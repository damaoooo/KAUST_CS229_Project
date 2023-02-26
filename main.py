import pytorch_lightning as pl
import torch
from model import ShortLanguageModel
from dataset import TOEFLDataModule
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    model_path = "albert-large-v2"
    my_model = ShortLanguageModel(lr=1e-5, model_path=model_path)

    my_dataset = TOEFLDataModule(batch_size=1, windows=4, use_cache="", tokenizer=model_path)
    my_dataset.prepare_data()

    trainer = pl.Trainer(precision=16, accelerator="gpu", max_epochs=2)
    trainer.fit(model=my_model, train_dataloaders=my_dataset)
