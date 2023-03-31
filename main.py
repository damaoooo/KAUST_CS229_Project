import torch
from lightning.pytorch import Trainer

from model import LanguageModel
from dataset import CLOTHDataModule
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64"

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    model_path = "albert-xxlarge-v2"
    my_model = LanguageModel(lr=5e-6, model_path=model_path)

    my_dataset = CLOTHDataModule(batch_size=1, use_cache="./.data_cache", tokenizer=model_path)
    my_dataset.prepare_data()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        # strategy="deepspeed_stage_3_offload",
        precision=16,
        max_epochs=100,
        val_check_interval=0.1
    )
    trainer.fit(model=my_model, train_dataloaders=my_dataset)
