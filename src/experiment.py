from pydantic import BaseModel
import torch.nn as nn

import torch
from torch.utils.data import DataLoader


class ExperimentResults(BaseModel):
    train_loss: list = []
    train_acc: list = []
    valid_loss: list = []
    valid_acc: list = []


class Experiment:

    def __init__(self, model: nn.Module, optimizer, criterion, device) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        grad_clip=None,
        max_lr=0.05,
        **kwargs
    ) -> ExperimentResults:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            epochs=epochs,
            max_lr=max_lr,
            steps_per_epoch=len(train_loader),
            **kwargs,
        )

        results = ExperimentResults()

        for i in range(epochs):
            loss, acc = self._train_epoch(loader=train_loader, grad_clip=grad_clip)
            results.train_loss.append(loss)
            results.train_acc.append(acc)

            loss, acc = self.evaluate(loader=valid_loader)
            results.valid_loss.append(loss)
            results.valid_acc.append(acc)

            print(
                f"Epoch {i + 1:02d}/{epochs} | "
                f"train_loss={results.train_loss[-1]:.4f} | "
                f"train_acc={results.train_acc[-1]:.4f} | "
                f"valid_loss={results.valid_loss[-1]:.4f} | "
                f"valid_acc={results.valid_acc[-1]:.4f}"
            )

        return results

    def _train_epoch(self, loader: DataLoader, grad_clip) -> tuple[float, float]:
        self.model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0

        for X, Y in loader:
            images, labels = X.to(self.device), Y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_acc += (outputs.argmax(dim=1) == labels).sum().item()  # adding up the n_correct

        avg_loss = epoch_loss / len(loader.dataset)
        avg_acc = epoch_acc / len(loader.dataset)

        return avg_loss, avg_acc

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        self.model.to(self.device)

        eval_loss = 0.0
        eval_acc = 0.0

        with torch.no_grad():
            for X, Y in loader:
                images, labels = X.to(self.device), Y.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                eval_loss += loss.item() * images.size(0)
                eval_acc += (outputs.argmax(dim=1) == labels).sum().item()

        avg_loss = eval_loss / len(loader.dataset)
        avg_acc = eval_acc / len(loader.dataset)

        return avg_loss, avg_acc
