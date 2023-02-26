import torch
import transformers

from utils import *
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel


class Classifier(nn.Module):
    def __init__(self, input_size=512, num_layers=4, hidden_size=512):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_layers,
                            bidirectional=True)
        self.label = nn.Sequential(*[
            nn.ReLU(),
            nn.Linear(2 * self.hidden_size * 12, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        ])

    def forward(self, x: torch.Tensor):
        batch_size, sequence, hidden = x.shape
        # ---------------------------
        #   [Batch_size x 16  x embedding]
        # ___________________________
        x, _ = self.lstm(x)
        # ---------------------------
        #   [Batch_size x 16 x embedding * 2]
        # ___________________________
        x = x.reshape(batch_size, -1)
        # ---------------------------
        #   [Batch_size x 16 * embedding * 2]
        # ___________________________
        x = self.label(x)
        return x


class ShortLanguageModel(pl.LightningModule):
    def __init__(self, model_path: str = "bert-base-cased", lr: float = 1e-3):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.lr = lr
        self.classifier = Classifier(input_size=self.model.config.hidden_size)
        self.save_hyperparameters()

    def forward(self, corpus: torch.Tensor, attn_mask: torch.Tensor):
        # --------------------------
        # [batch size x 16 x seq_len]
        # --------------------------

        x = corpus
        batch_size, sequence, sentences = x.shape
        x = x.reshape(batch_size * sequence, sentences)
        attn_mask = attn_mask.reshape(batch_size * sequence, sentences)
        # --------------------------
        # [batch size * 16 x seq_len]
        # --------------------------
        x = self.model(input_ids=x, attention_mask=attn_mask).pooler_output
        # ---------------------------
        #   [Batch_size * 16  x embedding]
        # ___________________________
        x = x.reshape(batch_size, sequence, -1)
        # ---------------------------
        #   [Batch_size x 16  x embedding]
        # ___________________________
        x = self.classifier.forward(x)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        corpus = batch['corpus']
        attn_mask = batch['mask']
        answer = batch['answer']
        output = self.forward(corpus, attn_mask)
        loss = F.cross_entropy(output, answer)
        with torch.no_grad():
            batch_size = answer.shape
            predict = torch.argmax(output, dim=-1)
            accuracy = (predict == answer).sum() / batch_size[0]
            predict = predict[0]
            answer = answer[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_predict", predict, on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        self.log("train_answer", answer, on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        corpus = batch['corpus']
        attn_mask = batch['mask']
        answer = batch['answer']
        output = self.forward(corpus, attn_mask)
        loss = F.cross_entropy(output, answer)
        batch_size = answer.shape
        accuracy = (torch.argmax(output, dim=-1) == answer).sum() / batch_size[0]
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", accuracy, sync_dist=True)
        return loss


if __name__ == '__main__':
    model = ShortLanguageModel()
