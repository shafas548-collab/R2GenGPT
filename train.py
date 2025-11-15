import os
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.R2GenGPT import R2GenGPT
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import warnings
import transformers

# 🔇 Matikan warning umum
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 🔇 Matikan semua log Hugging Face (transformers)
transformers.utils.logging.set_verbosity_error()


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
