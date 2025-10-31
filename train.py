import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.R2GenGPT import R2GenGPT
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torch


def build_model(args):
    # kalau mau load ckpt, tetap panggil ctor kita dulu, baru load state_dict
    if args.ckpt_file is not None:
        # 1) buat model dgn args → supaya semua logic __init__ jalan (vision, llama, proj, dll.)
        model = R2GenGPT(args)

        # 2) load state_dict ke model ini
        ckpt = torch.load(args.ckpt_file, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"[train.py] loaded checkpoint from {args.ckpt_file}")
        return model
    else:
        return R2GenGPT(args)


def train(args):
    seed_everything(42, workers=True)

    # 1) bangun model DULU
    model = build_model(args)

    # 2) baru bangun datamodule
    dm = DataModule(args)

    # 3) callback & logger
    cb = add_callbacks(args)

    # 4) trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,   # "gpu"
        devices=args.devices,           # 2
        num_nodes=args.num_nodes,
        strategy=args.strategy,         # "ddp"
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=cb["callbacks"],
        logger=cb["loggers"],
    )

    # 5) run
    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)


def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    train(args)


if __name__ == "__main__":
    main()
