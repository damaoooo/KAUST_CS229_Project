import torch
from lightning.pytorch import Trainer

from model import ShortLanguageModel
from dataset import TOEFLDataModule
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    model_path = "albert-base-v2"
    my_model = ShortLanguageModel(lr=1e-6, model_path=model_path)

    my_dataset = TOEFLDataModule(batch_size=1, windows=4, use_cache="", tokenizer=model_path)
    my_dataset.prepare_data()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        strategy="deepspeed_stage_3_offload",
        precision=16,
        max_epochs=1000
    )
    trainer.fit(model=my_model, train_dataloaders=my_dataset)
