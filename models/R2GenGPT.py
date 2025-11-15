import os
import json
import torch
import torch.serialization
import torch.nn as nn
import pytorch_lightning as pl
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    SwinModel,
)
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
from peft import LoraConfig, get_peft_model
import pdb

class R2GenGPT(pl.LightningModule):
    """
    DDP-safe version.
    - TIDAK menyimpan self.embed_tokens permanen (karena bisa beda device per rank)
    - SETIAP forward ambil ulang dari self.llama_model.get_input_embeddings()
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # -------------------------------------------------
        # 1) Vision encoder
        # -------------------------------------------------
        print(f"[R2GenGPT] Loading vision encoder: {args.vision_model}")
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print("[R2GenGPT] Vision encoder with LoRA -- Done")
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')
        print('Loading LLAMA model...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0

        # ============================================================
        # 🔹 Case 1: Low-resource mode → 4-bit + QLoRA
        # ============================================================
        if args.low_resource:
            print("[R2GenGPT] Low-resource: load 4-bit + QLoRA")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=None,   # ❌ jangan "auto" (DDP unsafe)
                low_cpu_mem_usage=True
            )

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.llm_lora_dropout,
                bias="none",
                target_modules=["q_proj", "v_proj"]  # standar untuk LLAMA
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print("Loading 4-bit QLoRA LLAMA Done ✅")
            
        # ============================================================
        # 🔹 Case 2: Fake QAT + Freeze
        # ============================================================
            
        elif args.fake_qat:
            print("→ QAT-Fake mode detected: LLaMA frozen, fake quant active")

            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                device_map=None,
            )

            # Freeze semua parameter (LLM tidak dilatih)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

            # Siapkan konfigurasi quantization aware training (fake quantization)
            qat_qconfig = get_default_qat_qconfig("fbgemm")
            self.llama_model.qconfig = qat_qconfig
            prepare_qat(self.llama_model, inplace=True)

            self.embed_tokens = self.llama_model.get_input_embeddings()
            print("✅ Fake QAT prepared (simulated 8-bit quantization, FP16 compute)")

        # ============================================================
        # 🔹 Case 2: Full mode → FP16 (no quantization, no LoRA)
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
            print("Loading FP16 LLAMA Done ✅")

        # ============================================================
        # Linear projection for visual features → LLAMA space
        # ============================================================
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)

        # prompt & buffer
        self.end_sym = args.end_sym
        self.prompt = (
            "Generate a comprehensive and detailed diagnosis report for this chest xray image."
        )
        self.val_step_outputs = []
        self.test_step_outputs = []
        self._epoch_vram = []
        self._epoch_util = []
        self.test_latencies = []
        self.test_utils = []
        self.test_vrams = []
        self.val_score = 0.0

        # delta
        if args.delta_file is not None:
            # ✅ Kompatibel dengan semua versi Lightning (lama & baru)
            try:
                from lightning_fabric.utilities.data import AttributeDict  # Lightning >= 2.0
            except ImportError:
                from pytorch_lightning.utilities.data import AttributeDict  # Lightning < 2.0

            # Izinkan AttributeDict supaya tidak error saat load
            torch.serialization.add_safe_globals([AttributeDict])

            # ✅ Load checkpoint (PyTorch >= 2.6 perlu weights_only=False)
            state = torch.load(args.delta_file, map_location='cuda', weights_only=False)

            # Ambil state_dict model (fleksibel)
            state_dict = state.get('model', state)

            # Load ke model
            self.load_state_dict(state_dict, strict=False)
            print(f'✅ Loaded checkpoint from {args.delta_file}')



    def score(self, ref, hypo):
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr"),
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, _ = scorer.compute_score(ref, hypo)
            if isinstance(score, list):
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    # ============================================================
    # Image encoder (sudah 4D-safe)
    # ============================================================
    def encode_img(self, images):
        device = images.device

        # 5D → ambil yang pertama
        if images.dim() == 5:
            images = images[:, 0]
        # 3D → tambah batch
        if images.dim() == 3:
            images = images.unsqueeze(0)

        assert images.dim() == 4, f"expect 4D (B,C,H,W), got {images.shape}"

        if self.hparams.global_only:
            feats = self.visual_encoder(images)["pooler_output"].unsqueeze(1)
        else:
            feats = self.visual_encoder(images)["last_hidden_state"]

        feats = self.llama_proj(feats)
        atts = torch.ones(feats.size()[:-1], dtype=torch.long, device=device)
        return feats, atts

    # ============================================================
    # Prompt wrap (perbaikan utama: embed_tokens diambil ulang)
    # ============================================================
    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img


    # ============================================================
    # Forward (train)
    # ============================================================
    def forward(self, samples):
        device = next(self.parameters()).device

        # 1) image
        image = samples["image"]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        image = image.to(device)
        if image.dim() == 5 and image.size(1) == 1:
            image = image[:, 0]

        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        # 2) text
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        # 3) kosongkan target utk prompt + BOS
        empty_targets = torch.ones(
            (atts_img.size(0), atts_img.size(1) + 1),
            dtype=torch.long,
            device=device,
        ).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        # 4) bangun inputs
        bsz = img_embeds.size(0)
        bos = torch.full(
            (bsz, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        embed_tokens = self._get_embed_tokens(device)

        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat(
            [bos_embeds, img_embeds, to_regress_embeds], dim=1
        )
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1
        )

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

    # ============================================================
    # Lightning hooks
    # ============================================================
    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

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
        # 🔹 Buat folder checkpoints
        ckpt_dir = os.path.abspath(os.path.join(self.hparams.savedmodel_path, "weights"))
        os.makedirs(ckpt_dir, exist_ok=True)

        filename = f"checkpoint_epoch{current_epoch}_step{global_step}_bleu{eval_res['Bleu_4']:.3f}_cider{eval_res['CIDEr']:.3f}.pth"
        save_to = os.path.join(ckpt_dir, filename)

        # 🔹 Simpan checkpoint
        torch.save(save_obj, save_to)
        print(f"Checkpoint saved at step {global_step} → {save_to}")
        
    
    def validation_step(self, samples, batch_idx):
        device = next(self.parameters()).device

        # loss
        with torch.no_grad():
            out = self(samples)
            val_loss = out["loss"].detach()

        # generate (harus ulang supaya pakai bos + prompt)
        image = samples["image"]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        image = image.to(device)
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        bsz = img_embeds.size(0)
        bos = torch.full(
            (bsz, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        embed_tokens = self._get_embed_tokens(device)
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
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        text = text.split("</s>")[0].strip()
        text = text.replace("<unk>", "")
        return text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"

        # target
        to_regress_tokens = self.llama_tokenizer(
            samples["input_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(device)

        # image
        image = samples["image"]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        image = image.to(device)

        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        bsz = img_embeds.size(0)
        bos = torch.full(
            (bsz, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        embed_tokens = self._get_embed_tokens(device)
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
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        ref, hypo, ids = [], [], []
        for item in self.test_step_outputs:
            ref.extend(item["ref"])
            hypo.extend(item["hypo"])
            ids.extend(item["id"])

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, "result")
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, "test_result.json"), "w"))
        json.dump(ref, open(os.path.join(result_folder, "test_refs.json"), "w"))
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

    # ============================================================
    # Optimizer
    # ============================================================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ------------------------------------------------------------
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()