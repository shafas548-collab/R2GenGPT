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
    R2GenGPT model (DDP-safe version).
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # 1️⃣ Vision Encoder
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
            print("Vision encoder with LoRA ✅")
        elif args.freeze_vm:
            for _, p in self.visual_encoder.named_parameters():
                p.requires_grad = False
            print("Vision encoder frozen ✅")
        else:
            print("Vision encoder trainable ✅")

        # 2️⃣ Tokenizer
        print("Loading LLAMA model & tokenizer...")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0

        # 3️⃣ LLaMA Model (4-bit or full precision)
        if args.low_resource:
            print("Low-resource mode: loading 4-bit LLAMA...")
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

            print("Applying QLoRA...")
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
            print("Loaded LLAMA 4-bit + QLoRA ✅")
        else:
            print("Full precision LLAMA (FP16)...")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model, torch_dtype=torch.float16, device_map=None
            )
            for _, p in self.llama_model.named_parameters():
                p.requires_grad = False
            print("Loaded LLAMA FP16 ✅")

        # 4️⃣ Vision → LLaMA projection
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)

        # Prompt + misc
        self.end_sym = args.end_sym
        self.prompt = "Generate a comprehensive and detailed diagnosis report for this chest X-ray image."
        self.val_step_outputs, self.test_step_outputs = [], []
        self.val_score = 0.0

        # Optional: delta checkpoint
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location="cpu")["model"]
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded delta checkpoint: {args.delta_file}")

    # ============================================================
    # Image Encoder
    # ============================================================
    def encode_img(self, images):
        device = images.device
        if images.dim() == 5:
            images = images[:, 0]
        if images.dim() == 3:
            images = images.unsqueeze(0)
        assert images.dim() == 4, f"Expected (B,C,H,W), got {images.shape}"

        feats = (
            self.visual_encoder(images)["pooler_output"].unsqueeze(1)
            if self.hparams.global_only
            else self.visual_encoder(images)["last_hidden_state"]
        )
        feats = self.llama_proj(feats)
        atts = torch.ones(feats.size()[:-1], dtype=torch.long, device=device)
        return feats, atts

    # ============================================================
    # Prompt Wrapper (DDP-safe)
    # ============================================================
    def prompt_wrap(self, img_embeds, atts_img):
        device = img_embeds.device
        prompt = f"Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:"
        batch_size = img_embeds.size(0)
        p_before, p_after = prompt.split("<ImageHere>")

        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
        p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(device)

        embed_tokens = self.llama_model.get_input_embeddings().to(device)
        p_before_embeds = embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.size(1)).to(device)
        return wrapped_img_embeds, wrapped_atts_img

    # ============================================================
    # Forward (Train)
    # ============================================================
    def forward(self, samples):
        device = next(self.parameters()).device

        # Image
        image = samples["image"]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        image = image.to(device)
        if image.dim() == 5 and image.size(1) == 1:
            image = image[:, 0]

        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        # Text
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]
        to_regress_tokens = self.llama_tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.hparams.max_length, add_special_tokens=False
        ).to(device)

        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == 0, -100)
        empty_targets = torch.ones((atts_img.size(0), atts_img.size(1) + 1), dtype=torch.long, device=device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        # Inputs
        batch_size = img_embeds.size(0)
        bos = torch.full((batch_size, 1), self.llama_tokenizer.bos_token_id, dtype=torch.long, device=device)

        embed_tokens = self.llama_model.get_input_embeddings().to(device)
        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        to_regress_embeds = embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=targets)
        return {"loss": outputs.loss}

    # ============================================================
    # Training
    # ============================================================
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out["loss"]
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    # ============================================================
    # Validation
    # ============================================================
    def validation_step(self, samples, batch_idx):
        device = next(self.parameters()).device
        to_regress_tokens = self.llama_tokenizer(
            samples["input_text"], return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.hparams.max_length, add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            val_loss = self(samples)["loss"].detach()

        image = samples["image"]
        if isinstance(image, list):
            image = torch.stack(image, dim=0)
        image = image.to(device)
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.size(0)
        bos = torch.full((batch_size, 1), self.llama_tokenizer.bos_token_id, dtype=torch.long, device=device)
        embed_tokens = self.llama_model.get_input_embeddings().to(device)

        bos_embeds = embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                num_beams=self.hparams.beam_size, do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens, max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty, temperature=self.hparams.temperature,
            )

        hypo = [self.decode(o) for o in outputs]
        ref = [self.decode(o) for o in to_regress_tokens["input_ids"]]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"], "val_loss": val_loss})
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] in [0, 1]:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        return output_text.split("</s>")[0].replace("<unk>", "").strip()

    # ============================================================
    # Optimizer
    # ============================================================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
