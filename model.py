import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import transformers


class MyModelModule(pl.LightningModule):
    def __init__(self, model_path: str = "bert-base-uncased", learning_rate: float = 1e-5):
        super().__init__()
        self.model_path = model_path
        self.model = transformers.BertForNextSentencePrediction.from_pretrained(self.model_path)
        # self.model = torch.compile(self.model)
        self.learning_rate = learning_rate

        self.train_step_output = []
        self.val_step_output = []

    def forward(self, x):
        r1, r2, label = x
        logit1 = self.model(**r1).logits
        logit2 = self.model(**r2).logits

        x = (logit1 + logit2) / 2
        loss = F.cross_entropy(x, label)

        with torch.no_grad():
            ok = x.argmax(dim=-1)
            ok = torch.sum(x == label).item()

        return ok, loss

    def training_step(self, batch, batch_idx):
        batch_len = len(batch[0])
        ok, loss = self.forward(batch)
        self.train_step_output.append([batch_len, ok])

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", ok / batch_len, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        acc = np.sum([x[1] for x in self.train_step_output]) / np.sum([x[0] for x in self.train_step_output])
        self.log("train_acc", acc.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.train_step_output.clear()

    def validation_step(self, batch, batch_idx):
        batch_len = len(batch[0])
        loss, ok = self.forward(batch)
        self.log("val_loss", loss.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.val_step_output.append([batch_len, ok])
        return loss

    def on_validation_epoch_end(self):
        acc = np.sum([x[1] for x in self.val_step_output]) / np.sum([x[0] for x in self.val_step_output])
        self.log("val_acc", acc.item(), on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.val_step_output.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
