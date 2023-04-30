import torch
import os
from lightning.pytorch import Trainer
from model import MyModelModule
from dataset import SCDEDataModule

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64"
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    my_dataset = SCDEDataModule(batch_size=32, num_workers=8)

    my_model = MyModelModule(model_path="bert-base-uncased")

    trainer = Trainer(
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=100,
        # val_check_interval=0.2,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value"
    )
    # trainer.validate(model=my_model, dataloaders=my_dataset)
    trainer.fit(model=my_model, datamodule=my_dataset)
