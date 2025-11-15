import os
import json
import torch
import torch.serialization
from lightning_fabric.utilities.data import AttributeDict
import torch.nn as nn
import pytorch_lightning as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import pynvml, psutil, time, csv
pynvml.nvmlInit()

class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.vis_lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            trainable_params = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.visual_encoder.parameters())
            print(f"[Vision Encoder] trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            trainable_params = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.visual_encoder.parameters())
            print(f"[Vision Encoder] trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
            print(f'Loading Full Trainable vision encoder:{args.vision_model} -- Done')
        print('Loading LLAMA model...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        # ============================================================
        # 🔹 Case 1: Low-resource mode → 4-bit + QLoRA
        # ============================================================
        if args.low_resource:
            print("→ Low resource mode detected: loading 4-bit model with QLoRA")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # ⛔ DDP-safe: no device_map="auto"
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=None,   # ❌ jangan "auto" (DDP unsafe)
            )

            # ✅ Tambahkan LoRA (QLoRA)
            print("Applying QLoRA...")
            # keep embed_tokens attribute for backward compat but do not assume it's on-device
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.llm_lora_dropout,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print("Loading 4-bit QLoRA LLAMA Done ✅")
            
        # ============================================================
        # 🔹 Case 2: Fake QAT + Freeze (lm_head only)
        # ============================================================
        elif args.fake_qat:
            print("→ Fake QAT Simulasi INT8 aktif (lm_head only, clean mode)")

            # 1️⃣ Load model di FP16
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                device_map=None,
            )
            self.llama_model.eval()  # stabilisasi LayerNorm dan Dropout

            # 2️⃣ Freeze semua parameter (hemat & aman)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            # 3️⃣ Fake quantisasi simulasi pada lm_head
            w = self.llama_model.lm_head.weight.data
            qmin, qmax = -128, 127
            scale = w.abs().max() / max(qmax, 1)
            w_q = torch.clamp((w / scale).round(), qmin, qmax) * scale
            self.llama_model.lm_head.weight.data.copy_(w_q)
            print("✅ Fake QAT simulated on lm_head weights (INT8 emulation, no observer)")

            # 4️⃣ Ambil embedding (tetap simpan referensi)
            self.embed_tokens = self.llama_model.get_input_embeddings()
            print("✅ LLaMA frozen, Fake QAT ready (FP16 compute, INT8-like behavior)")

        
        # ============================================================
        # 🔹 Case 3: Full mode → FP16 (no quantization, no LoRA)
        # ============================================================
        else:
            print("→ Full precision mode detected: loading FP16 model")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                device_map=None  # DDP-safe
            )

            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
                
            trainable_params = sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.llama_model.parameters())
            print(f"[LLM] trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
            print("Loading FP16 LLAMA Done ✅")

        # ============================================================
        # Linear projection for visual features → LLAMA space
        # ============================================================
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self._epoch_vram = []
        self._epoch_util = []
        self._epoch_latencies = []
        self.test_latencies = []
        self.test_utils = []
        self.test_vrams = []
        self.val_score = 0.0

        if args.delta_file is not None:
            # Izinkan AttributeDict supaya tidak error saat load
            torch.serialization.add_safe_globals([AttributeDict])

            # ✅ Load checkpoint (PyTorch >= 2.6 perlu weights_only=False)
            state = torch.load(args.delta_file, map_location='cuda', weights_only=False)

            # Ambil state_dict model (fleksibel)
            state_dict = state.get('model', state)

            # Load ke model
            self.load_state_dict(state_dict, strict=False)
            print(f'✅ Loaded checkpoint from {args.delta_file}')
            
    # ---------------------------
    # helper: DDP-safe embed getter
    # ---------------------------
    def _get_embed_tokens(self, device):
        """
        Return the input embedding module moved (not copied) to 'device'.
        We return the module itself so calling code can do embed_tokens(input_ids).
        """
        # get the embedding layer from the underlying llama_model and ensure it's on the target device
        emb = self.llama_model.get_input_embeddings()
        try:
            emb.to(device)
        except Exception:
            # some embedding wrappers might not have .to implemented; rely on calling code to send input ids to device
            pass
        return emb

    def _log_gpu_cpu_epoch(self, prefix):
        """
        prefix: "train" atau "val"
        Mengumpulkan:
        - avg & peak VRAM
        - avg & peak GPU Util
        Mendukung:
        - single GPU
        - multi-GPU (auto adaptasi)
        """
    
        # ===================== Local GPU Stats =====================
        avg_vram_local = float(sum(self._epoch_vram) / max(1, len(self._epoch_vram)))
        peak_vram_local = float(max(self._epoch_vram)) if len(self._epoch_vram) > 0 else 0.0
    
        avg_util_local = float(sum(self._epoch_util) / max(1, len(self._epoch_util)))
        peak_util_local = float(max(self._epoch_util)) if len(self._epoch_util) > 0 else 0.0
    
        # convert to tensor
        t_avg_vram = torch.tensor(avg_vram_local, device=self.device)
        t_peak_vram = torch.tensor(peak_vram_local, device=self.device)
        t_avg_util = torch.tensor(avg_util_local, device=self.device)
        t_peak_util = torch.tensor(peak_util_local, device=self.device)
    
        # ===================== All Gather (multi-gpu safe) =====================
        all_avg_vram = self.all_gather(t_avg_vram).cpu().tolist()
        all_peak_vram = self.all_gather(t_peak_vram).cpu().tolist()
        all_avg_util = self.all_gather(t_avg_util).cpu().tolist()
        all_peak_util = self.all_gather(t_peak_util).cpu().tolist()
    
        # ===================== Logging =====================
        if self.trainer.is_global_zero:
    
            num_gpus = len(all_avg_vram)
    
            # log otomatis sesuai jumlah GPU (1, 2, 4, ...)
            for i in range(num_gpus):
                self.log(f"{prefix}_gpu{i}_avg_vram",
                         all_avg_vram[i], on_epoch=True, rank_zero_only=True)
    
                self.log(f"{prefix}_gpu{i}_peak_vram",
                         all_peak_vram[i], on_epoch=True, rank_zero_only=True)
    
                self.log(f"{prefix}_gpu{i}_avg_util",
                         all_avg_util[i], on_epoch=True, rank_zero_only=True)
    
                self.log(f"{prefix}_gpu{i}_peak_util",
                         all_peak_util[i], on_epoch=True, rank_zero_only=True)
    
            # CPU log
            mem = psutil.virtual_memory()
            self.log(f"{prefix}_cpu_ram",
                     mem.used / (1024 ** 3),
                     on_epoch=True,
                     rank_zero_only=True)
           

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores


    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            if self.hparams.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama


    def prompt_wrap(self, img_embeds, atts_img):
        """
        Ensure: p_before, img_embeds, and p_after have matching batch size and
        concatenate safely, padding where needed.
        """

        prompt = f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'

        # Split prompt
        p_before, p_after = prompt.split("<ImageHere>")
        batch = img_embeds.shape[0]
        device = img_embeds.device

        # Tokenize
        tok_before = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(device)
        tok_after = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False
        ).to(device)

        # Get embeddings
        before_emb = self.embed_tokens(tok_before.input_ids)     # [1, L1, 4096]
        after_emb = self.embed_tokens(tok_after.input_ids)       # [1, L2, 4096]

        # Expand batch
        before_emb = before_emb.expand(batch, -1, -1)            # [B, L1, 4096]
        after_emb = after_emb.expand(batch, -1, -1)              # [B, L2, 4096]

        # ----------------------------------------------------
        # FIX 🔥: Pastikan panjang img_embeds sama dalam batch
        # ----------------------------------------------------
        max_len = img_embeds.shape[1]
        if not all(img_embeds.shape[1] == max_len for _ in range(batch)):
            # pad semua image embeds agar sama panjang
            padded = []
            for i in range(batch):
                seq = img_embeds[i]
                pad_len = max_len - seq.shape[0]
                if pad_len > 0:
                    seq = torch.cat([seq, torch.zeros(pad_len, seq.shape[1], device=device)], dim=0)
                padded.append(seq.unsqueeze(0))
            img_embeds = torch.cat(padded, dim=0)

        # Wrap embeds
        wrapped = torch.cat([before_emb, img_embeds, after_emb], dim=1)

        # Attention mask
        wrapped_atts = torch.ones(wrapped.size()[:2], dtype=torch.long, device=device)

        return wrapped, wrapped_atts


    # ============================================================
    # Forward (train)
    # ============================================================
    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(img_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(img_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id

        # DDP-safe: ambil embed module di device sekarang
        embed_tokens = self._get_embed_tokens(img_embeds.device)

        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}
    
    def on_train_epoch_start(self):
        self._epoch_vram = []
        self._epoch_util = []
        self._epoch_latencies = []

    def training_step(self, batch, batch_idx):
        start = time.time()
        result = self(batch)
        step_latency = time.time() - start
        self._epoch_latencies.append(step_latency)
        self.log_dict(result, prog_bar=True)

        # track vram per-step
        device_idx = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

        vram = torch.cuda.memory_allocated() / (1024 ** 3)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

        self._epoch_vram.append(vram)
        self._epoch_util.append(util)

        return result
        
    def on_train_epoch_end(self):
        self._log_gpu_cpu_epoch("train")

        # ===== CSV Logging for Train =====
        avg_lat = float(np.mean(self._epoch_latencies)) if len(self._epoch_latencies) else 0
        std_lat = float(np.std(self._epoch_latencies)) if len(self._epoch_latencies) else 0
        throughput = 1.0 / avg_lat if avg_lat > 0 else 0.0

        avg_vram = float(np.mean(self._epoch_vram)) if len(self._epoch_vram) else 0
        peak_vram = float(np.max(self._epoch_vram)) if len(self._epoch_vram) else 0
        avg_util = float(np.mean(self._epoch_util)) if len(self._epoch_util) else 0
        peak_util = float(np.max(self._epoch_util)) if len(self._epoch_util) else 0

        csv_path = os.path.join(self.hparams.savedmodel_path, "latensi-train.csv")
        header = ["epoch", "avg_latency", "std_latency", "throughput",
                  "avg_util", "peak_util", "avg_vram", "peak_vram"]
        row = [self.current_epoch, avg_lat, std_lat, throughput,
               avg_util, peak_util, avg_vram, peak_vram]

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        self.print(f"[INFO] Saved train latency metrics → {csv_path}")

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'weights'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'weights',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
        
    def on_validation_epoch_start(self):
        self._epoch_vram = []
        self._epoch_util = []
        self._epoch_latencies = []
        
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id

        # DDP-safe embed getter
        embed_tokens = self._get_embed_tokens(img_embeds.device)

        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        # timing around generate
        start = time.time()
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        step_latency = time.time() - start
        self._epoch_latencies.append(step_latency)

        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        
        device_idx = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

        vram = torch.cuda.memory_allocated() / (1024 ** 3)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

        self._epoch_vram.append(vram)
        self._epoch_util.append(util)

        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        
        # 🔄 Sinkron antar GPU (tanpa all_reduce manual)
        for k, v in eval_res.items():
            eval_res[k] = self.trainer.strategy.reduce(v, reduce_op="mean")
            
        self.log_dict(eval_res, sync_dist=False, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        # ⬇️ Tambahkan rank di nama file agar tidak tabrakan
        rank = self.global_rank if hasattr(self, "global_rank") else 0
        json.dump(hypo, open(os.path.join(result_folder, f"result_rank{rank}_{current_epoch}_{global_step}.json"), "w"))
        json.dump(ref, open(os.path.join(result_folder, f"refs_rank{rank}.json"), "w"))
        self.print(eval_res)

        val_score = 0.0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        # 🔄 Sinkronkan val_score antar GPU (mean semua rank)
        val_score = self.trainer.strategy.reduce(val_score, reduce_op="mean")

        # 💾 Simpan checkpoint hanya di rank 0, tapi berdasarkan val_score global
        if self.trainer.is_global_zero:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score

        # 🧹 Bersihkan buffer
        self.val_step_outputs.clear()
        self._log_gpu_cpu_epoch("val")

        # ===== CSV Logging for Validation =====
        avg_lat = float(np.mean(self._epoch_latencies)) if len(self._epoch_latencies) else 0
        std_lat = float(np.std(self._epoch_latencies)) if len(self._epoch_latencies) else 0
        throughput = 1.0 / avg_lat if avg_lat > 0 else 0.0

        avg_vram = float(np.mean(self._epoch_vram)) if len(self._epoch_vram) else 0
        peak_vram = float(np.max(self._epoch_vram)) if len(self._epoch_vram) else 0
        avg_util = float(np.mean(self._epoch_util)) if len(self._epoch_util) else 0
        peak_util = float(np.max(self._epoch_util)) if len(self._epoch_util) else 0

        csv_path = os.path.join(self.hparams.savedmodel_path, "latensi-val.csv")
        header = ["epoch", "avg_latency", "std_latency", "throughput",
                  "avg_util", "peak_util", "avg_vram", "peak_vram"]
        row = [current_epoch, avg_lat, std_lat, throughput,
               avg_util, peak_util, avg_vram, peak_vram]

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        self.print(f"[INFO] Saved validation latency metrics → {csv_path}")
        
        
    def test_step(self, samples, batch_idx):
        start = time.time()
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id

        # DDP-safe embed getter
        embed_tokens = self._get_embed_tokens(img_embeds.device)

        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature, 
        )
        # --- timing end ---
        end = time.time()
        latency = end - start

        # ===== GPU UTIL + VRAM =====
        device_idx = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3

        # ===== Simpan untuk perhitungan final =====
        self.test_latencies.append(latency)
        self.test_utils.append(util)
        self.test_vrams.append(vram)
        
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")
        # ======== PERFORMANCE LOGGING =========
        avg_lat = float(np.mean(self.test_latencies))
        std_lat = float(np.std(self.test_latencies))
        throughput = 1.0 / avg_lat if avg_lat > 0 else 0.0
        avg_util = float(np.mean(self.test_utils))
        peak_util = float(np.max(self.test_utils))
        avg_vram = float(np.mean(self.test_vrams))
        peak_vram = float(np.max(self.test_vrams))

        csv_path = os.path.join(self.hparams.savedmodel_path, "latensi-test.csv")
        header = [
            "avg_latency", "std_latency", "throughput",
            "avg_util", "peak_util", "avg_vram", "peak_vram"
        ]
        data = [avg_lat, std_lat, throughput,
                avg_util, peak_util, avg_vram, peak_vram]

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data)
        self.print(f"[INFO] Saved latency metrics → {csv_path}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
