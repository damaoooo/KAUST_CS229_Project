import torch
from lightning.pytorch import Trainer

from model import LanguageModel
from dataset import CLOTHDataModule
import os
import torch.multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64"


torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    model_path = "bert-base-uncased"
    my_model = LanguageModel(lr=5e-6, model_path=model_path)

    my_dataset = CLOTHDataModule(batch_size=1, use_cache="./.data_cache", tokenizer=model_path)
    my_dataset.prepare_data()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        strategy="ddp",
        # strategy="deepspeed_stage_3_offload",
        precision="16-mixed",
        max_epochs=500
    )
    trainer.fit(model=my_model, train_dataloaders=my_dataset)
