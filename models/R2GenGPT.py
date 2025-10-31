import os
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
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
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
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
                device_map=None,  
                low_cpu_mem_usage=True
            )

            # ✅ Tambahkan LoRA (QLoRA)
            print("Applying QLoRA...")
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=["q_proj", "v_proj"]  # standar untuk LLAMA
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print("Loading 4-bit QLoRA LLAMA Done ✅")

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
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


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
    device = img_embeds.device
    prompt = f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
    batch_size = img_embeds.shape[0]

    p_before, p_after = prompt.split('<ImageHere>')
    p_before_tokens = self.llama_tokenizer(
        p_before, return_tensors="pt", add_special_tokens=False
    ).to(device)
    p_after_tokens = self.llama_tokenizer(
        p_after, return_tensors="pt", add_special_tokens=False
    ).to(device)

    p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
    p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

    wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
    wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
    return wrapped_img_embeds, wrapped_atts_img



    def forward(self, samples):
    # 1) ambil device tempat model ini lagi jalan (cuda:0 di rank 0, cuda:1 di rank 1)
    device = next(self.parameters()).device

    # 2) pindahin image ke device itu
    image = samples["image"].to(device)

    # 3) encode image seperti biasa
    img_embeds, atts_img = self.encode_img(image)
    img_embeds = self.layer_norm(img_embeds)

    # 4) bungkus prompt — di prompt_wrap juga nanti semua kita paksa ke device ini
    img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

    # 5) siapkan teks target
    self.llama_tokenizer.padding_side = "right"
    text = [t + self.end_sym for t in samples["input_text"]]

    to_regress_tokens = self.llama_tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=self.hparams.max_length,
        add_special_tokens=False
    ).to(device)   # <— penting: ke device yg sama

    # 6) bikin target
    targets = to_regress_tokens.input_ids.masked_fill(
        to_regress_tokens.input_ids == 0, -100
    )

    empty_targets = (
        torch.ones(
            [atts_img.shape[0], atts_img.shape[1] + 1],
            dtype=torch.long,
            device=device,
        ).fill_(-100)
    )
    targets = torch.cat([empty_targets, targets], dim=1)

    # 7) BOS di device yg sama
    batch_size = img_embeds.shape[0]
    bos = torch.ones(
        [batch_size, 1],
        dtype=to_regress_tokens.input_ids.dtype,
        device=device,
    ) * self.llama_tokenizer.bos_token_id
    bos_embeds = self.embed_tokens(bos)
    atts_bos = atts_img[:, :1]

    # 8) gabungkan semua embed
    to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
    inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
    attention_mask = torch.cat(
        [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1
    )

    # 9) forward ke LLaMA
    outputs = self.llama_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        return_dict=True,
        labels=targets,
    )
    loss = outputs.loss
    return {"loss": loss}


    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log("train_loss", result["loss"], prog_bar=True, on_step=False, on_epoch=False, logger=False)
        return result

    def on_train_epoch_end(self):
        epoch_loss = torch.stack(
            [x['loss'] for x in self.trainer.fit_loop.epoch_loop.batch_loop.outputs]
        ).mean()
        
        current_step = self.trainer.global_step  # ambil step terakhir dari epoch ini
        self.log("train_epoch_loss", epoch_loss, on_epoch=True, on_step=False, logger=True, prog_bar=False)
        self.logger.log_metrics({"train_epoch_loss": epoch_loss.item()}, step=current_step)

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
        ckpt_dir = os.path.join(self.hparams.savedmodel_path, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        filename = f"checkpoint_epoch{current_epoch}_step{global_step}_bleu{eval_res['Bleu_4']:.3f}_cider{eval_res['CIDEr']:.3f}.pth"
        save_to = os.path.join(ckpt_dir, filename)

        # 🔹 Simpan checkpoint
        self.print(f"💾 Saving checkpoint at step {global_step} → {save_to}")
        torch.save(save_obj, save_to)
        self.print("✅ Save checkpoint -- Done")
        
    
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
        
        # 🔹 Hitung loss juga untuk val_loss
        with torch.no_grad():
            outputs_loss = self(samples)  # gunakan forward() yg sudah ada
            val_loss = outputs_loss["loss"].detach()

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
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
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"], "val_loss": val_loss})
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
        ref, hypo, ids, val_losses = [], [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])
            val_losses.append(i['val_loss'])
        
        # 🔹 Rata-rata val_loss untuk epoch
        val_epoch_loss = torch.stack(val_losses).mean()
        current_step = self.trainer.global_step
        self.log("val_epoch_loss", val_epoch_loss, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.logger.log_metrics({"val_epoch_loss": val_epoch_loss.item()}, step=current_step)

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, on_step=False, on_epoch=True, sync_dist=True, logger=True, prog_bar=False)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

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
        bos_embeds = self.embed_tokens(bos)
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()