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
from peft import get_peft_model, LoraConfig, TaskType


class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model (DDP-safe, follow-embedding-device).
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # 1) Vision encoder
        print(f"Loading vision encoder: {args.vision_model}")
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
            print("Vision encoder with LoRA -- done")
        elif args.freeze_vm:
            for _, p in self.visual_encoder.named_parameters():
                p.requires_grad = False
            print("Vision encoder frozen -- done")
        else:
            print("Vision encoder trainable -- done")

        # 2) Tokenizer
        print("Loading LLaMA tokenizer...")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(
            args.llama_model, use_fast=False
        )
        self.llama_tokenizer.pad_token_id = 0

        # 3) LLaMA model
        print("Loading LLaMA model...")
        if args.low_resource:
            print("→ Low resource (4-bit + QLoRA)")
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
                device_map=None,
                low_cpu_mem_usage=True,
            )
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=["q_proj", "v_proj"],
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print("LLaMA 4-bit + QLoRA -- done ✅")
        else:
            print("→ Full precision (fp16) + frozen")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                device_map=None,
            )
            for _, p in self.llama_model.named_parameters():
                p.requires_grad = False
            print("LLaMA fp16 frozen -- done ✅")

        # 4) Vision → LLaMA proj
        self.llama_proj = nn.Linear(
            self.visual_encoder.num_features,
            self.llama_model.config.hidden_size,
        )
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)

        # Prompt & buffers
        self.end_sym = args.end_sym
        self.prompt = "Generate a comprehensive and detailed diagnosis report for this chest xray image."
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location="cpu")["model"]
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded delta file from {args.delta_file}")

    # =========================================================
    # metrics
    # =========================================================
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

    # =========================================================
    # encode_img
    # =========================================================
    def encode_img(self, images):
        # normalize shape to (B, C, H, W)
        device = images.device
        if images.dim() == 5:   # (B,1,C,H,W) or (B,N,C,H,W)
            images = images[:, 0]
        if images.dim() == 3:   # (C,H,W)
            images = images.unsqueeze(0)
        assert images.dim() == 4, f"expected 4D image, got {images.shape}"

        if self.hparams.global_only:
            feats = self.visual_encoder(images)["pooler_output"].unsqueeze(1)
        else:
            feats = self.visual_encoder(images)["last_hidden_state"]

        feats = self.llama_proj(feats)
        atts = torch.ones(feats.size()[:-1], dtype=torch.long, device=device)
        return feats, atts

    # =========================================================
    # prompt_wrap (here was your error)
    # =========================================================
    def prompt_wrap(self, img_embeds, atts_img):
        """
        PENTING: kita ikuti device-nya embedding LLaMA, bukan device kita.
        """
        # ambil embedding + device aslinya
        embed_tokens = self.llama_model.get_input_embeddings()
        emb_device = embed_tokens.weight.device

        # pindahkan image prompt ke device embedding
        img_embeds = img_embeds.to(emb_device)
        atts_img = atts_img.to(emb_device)

        prompt = f"Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:"
        bsz = img_embeds.size(0)
        p_before, p_after = prompt.split("<ImageHere>")

        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(emb_device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False
        ).to(emb_device)

        p_before_embeds = embed_tokens(p_before_tokens.input_ids).expand(bsz, -1, -1)
        p_after_embeds = embed_tokens(p_after_tokens.input_ids).expand(bsz, -1, -1)

        wrapped = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts = atts_img[:, :1].expand(-1, wrapped.size(1)).to(emb_device)
        return wrapped, wrapped_atts

    # =========================================================
    # forward (train)
    # =========================================================
    def forward(self, samples):
        # ambil image dari batch
        image = samples["image"]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)

        # untuk encode_img kita cukup kirim ke device view ini
        # nanti prompt_wrap akan mindahin ke device embedding
        image = image.to(next(self.visual_encoder.parameters()).device)

        # 1) encode image → lalu wrap
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        # 2) siapkan text target
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        # ambil device embedding sekali lagi (paling aman)
        embed_tokens = self.llama_model.get_input_embeddings()
        emb_device = embed_tokens.weight.device

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(emb_device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        # target kosong buat prompt + BOS
        empty_targets = torch.ones(
            (atts_img.size(0), atts_img.size(1) + 1),
            dtype=torch.long,
            device=emb_device,
        ).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        # 3) bangun inputs
        bsz = img_embeds.size(0)
        bos = torch.full(
            (bsz, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=emb_device,
        )

        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        to_regress_embeds = embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat(
            [bos_embeds, img_embeds, to_regress_embeds], dim=1
        )
        attention_mask = torch.cat(
            [atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1
        )

        # 4) forward ke LLaMA (di device embedding)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    # =========================================================
    # training_step
    # =========================================================
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out["loss"]
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    # =========================================================
    # validation_step
    # =========================================================
    def validation_step(self, samples, batch_idx):
        # pakai forward → otomatis ikut device embedding
        with torch.no_grad():
            val_out = self(samples)
            val_loss = val_out["loss"].detach()

        # sekarang generate caption
        # ambil device embedding
        embed_tokens = self.llama_model.get_input_embeddings()
        emb_device = embed_tokens.weight.device

        # text ref
        to_regress_tokens = self.llama_tokenizer(
            samples["input_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(emb_device)

        # image
        image = samples["image"]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        # encode (ke device vision dulu)
        image = image.to(next(self.visual_encoder.parameters()).device)
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        # wrap → pindah ke emb_device
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        bsz = img_embeds.size(0)
        bos = torch.full(
            (bsz, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=emb_device,
        )
        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=self.hparams.temperature,
            )

        hypo = [self.decode(o) for o in outputs]
        ref = [self.decode(o) for o in to_regress_tokens["input_ids"]]

        self.val_step_outputs.append(
            {
                "hypo": hypo,
                "ref": ref,
                "id": samples["id"],
                "val_loss": val_loss,
            }
        )
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] in (0, 1):
            output_token = output_token[1:]
        text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        text = text.split("</s>")[0].replace("<unk>", "").strip()
        return text

    # =========================================================
    # on_validation_epoch_end
    # =========================================================
    def on_validation_epoch_end(self):
        ref, hypo, ids, val_losses = [], [], [], []
        for item in self.val_step_outputs:
            ref.extend(item["ref"])
            hypo.extend(item["hypo"])
            ids.extend(item["id"])
            val_losses.append(item["val_loss"])

        if len(val_losses) > 0:
            val_epoch_loss = torch.stack(val_losses).mean()
        else:
            # rank yg ga dpt batch
            val_epoch_loss = torch.tensor(
                0.0, device=next(self.llama_model.parameters()).device
            )

        self.log(
            "val_epoch_loss",
            val_epoch_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}

        if self.trainer.is_global_zero:
            eval_res = self.score(ref=ref, hypo=hypo)

            if self.logger is not None:
                self.logger.log_metrics(
                    {k: float(v) for k, v in eval_res.items()},
                    step=self.trainer.global_step,
                )

            result_folder = os.path.join(self.hparams.savedmodel_path, "result")
            os.makedirs(result_folder, exist_ok=True)
            cur_epoch = self.trainer.current_epoch
            gstep = self.trainer.global_step

            json.dump(
                hypo,
                open(
                    os.path.join(
                        result_folder,
                        f"result_{cur_epoch}_{gstep}.json",
                    ),
                    "w",
                ),
            )
            json.dump(ref, open(os.path.join(result_folder, "refs.json"), "w"))
            self.print(eval_res)

            # pilih skor terbaik
            val_score = 0
            for score_type, weight in zip(
                self.hparams.scorer_types, self.hparams.weights
            ):
                val_score += eval_res[score_type] * weight

            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score

        self.val_step_outputs.clear()

    # =========================================================
    # test_step
    # =========================================================
    def test_step(self, samples, batch_idx):
        embed_tokens = self.llama_model.get_input_embeddings()
        emb_device = embed_tokens.weight.device

        to_regress_tokens = self.llama_tokenizer(
            samples["input_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(emb_device)

        image = samples["image"]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        image = image.to(next(self.visual_encoder.parameters()).device)
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)  # to emb_device

        bsz = img_embeds.size(0)
        bos = torch.full(
            (bsz, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=emb_device,
        )
        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=self.hparams.temperature,
            )

        hypo = [self.decode(o) for o in outputs]
        ref = [self.decode(o) for o in to_regress_tokens["input_ids"]]

        self.test_step_outputs.append(
            {"hypo": hypo, "ref": ref, "id": samples["id"]}
        )
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
        self.print(f"Test result: {eval_res}")

    # =========================================================
    # optimizer
    # =========================================================
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    # =========================================================
    # save checkpoint
    # =========================================================
    def save_checkpoint(self, eval_res):
        cur_epoch = self.trainer.current_epoch
        gstep = self.trainer.global_step

        grad_params = {k: v.requires_grad for k, v in self.named_parameters()}
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in grad_params or not grad_params[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": cur_epoch,
            "step": gstep,
        }

        ckpt_dir = os.path.join(self.hparams.savedmodel_path, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        fname = (
            f"checkpoint_epoch{cur_epoch}_step{gstep}_"
            f"bleu{eval_res['Bleu_4']:.3f}_cider{eval_res['CIDEr']:.3f}.pth"
        )
        save_to = os.path.join(ckpt_dir, fname)
        self.print(f"💾 Saving checkpoint to {save_to}")
        torch.save(save_obj, save_to)
        self.print("✅ Save checkpoint -- done")