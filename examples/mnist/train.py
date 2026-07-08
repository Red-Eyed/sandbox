from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lit_model import LitMNISTNet, MNISTDataModule

HERE = Path(__file__).parent


def main() -> None:
    data = MNISTDataModule(data_dir=str(HERE / "data"))
    model = LitMNISTNet()

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(HERE / "checkpoints"),
        filename="mnist_net",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
    )

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=data)
    trainer.validate(model, datamodule=data)

    print(f"Checkpoint saved to: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
