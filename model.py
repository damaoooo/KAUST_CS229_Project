import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
import lightning.pytorch as pl
import torch.nn.functional as F


class LanguageModel(pl.LightningModule):
    def __init__(self, model_path: str = "bert-base-uncased", lr: float = 1e-5):
        super().__init__()
        self.model_path = model_path
        self.lm_model = AutoModelForMaskedLM.from_pretrained(self.model_path)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, p):
        attn_mask = p["attn_mask"]
        length = p["length"]
        answer = p["answer"]
        article = p["article"]
        options = p["options"]
        position = p["position"]
        x: torch.Tensor = self.lm_model(article, attention_mask=attn_mask).logits
        # ----------------------------------------
        #      batch_size x seq_len x vocab_size
        # ----------------------------------------
        loss_each = []
        # ----------------------------------------
        #  x = batch_size x seq_len x vocab_size
        # ----------------------------------------
        acc_all = []
        for passage in range(len(x)):
            wanted = x[passage].index_select(dim=0, index=position[passage])
            wanted = wanted.index_select(dim=1, index=options[passage])
            # ----------------------------------------
            #  answer = batch_size x 20 wanted = 20 x 4
            # ----------------------------------------
            loss_each.append(F.cross_entropy(wanted[:length[passage]], answer[passage][:length[passage]]))
            with torch.no_grad():
                answer_each = torch.argmax(wanted, -1)[:length[passage]]
                acc_all = torch.sum((answer_each == answer[passage][:length[passage]])).item()
        loss_each = torch.stack(loss_each, -1)
        loss_each = torch.mean(loss_each)
        return torch.mean(loss_each), acc_all

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch)

        with torch.no_grad():
            total_count = 0
            total_acc = 0
            for b in range(len(batch['length'])):
                total_count += batch['length'][b]
                total_acc += acc[b]
            self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            self.log("train_acc", (total_acc/total_count), on_step=True, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_index):
        loss, acc = self.forward(batch)

        with torch.no_grad():
            total_count = 0
            total_acc = 0
            for b in range(len(batch['length'])):
                total_count += batch['length'][b]
                total_acc += acc[b]
            self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            self.log("val_acc", (total_acc / total_count), on_step=True, on_epoch=True, logger=True, sync_dist=True,
                     prog_bar=True)

        return loss


if __name__ == "__main__":
    model = LanguageModel()



