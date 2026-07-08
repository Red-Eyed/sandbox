import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from model import MNISTNet


class LitMNISTNet(L.LightningModule):
    """Lightning training wrapper around the plain MNISTNet.

    Kept separate from MNISTNet itself so the quantization pipeline only ever sees a
    plain nn.Module — it has no notion of Lightning's training-loop machinery.
    """

    def __init__(self, lr: float = 1e-3, num_classes: int = 10) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.net = MNISTNet(num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_accuracy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 128) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        from torchvision.datasets import MNIST

        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        from torchvision import transforms
        from torchvision.datasets import MNIST

        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_set = MNIST(self.data_dir, train=True, transform=tfm)
        self.val_set = MNIST(self.data_dir, train=False, transform=tfm)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)
