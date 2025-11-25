import os
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping



def add_callbacks(args):
log_dir = args.savedmodel_path
os.makedirs(log_dir, exist_ok=True)

# --------- Add Callbacks
checkpoint_callback = ModelCheckpoint(
dirpath=os.path.join(log_dir, "weights"),
filename="{epoch}-{step}",
save_top_k=-1,
every_n_train_steps=args.every_n_train_steps,
save_last=False,
save_weights_only=False
)

lr_monitor_callback = LearningRateMonitor(logging_interval='step')
tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    early_stop_callback = EarlyStopping(
        monitor="Bleu_4",
        patience=8,
        mode="max"
    )

to_returns = {
        "callbacks": [checkpoint_callback, lr_monitor_callback],
        "callbacks": [checkpoint_callback, lr_monitor_callback, early_stop_callback],
"loggers": [csv_logger, tb_logger]
}

return to_returns