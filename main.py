from lightning_module import CLIP
from pairdataset import PokemonClipDataset
import pytorch_lightning as pl
import torch 
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import os
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

NUM_HEADS = 8
NUM_LAYERS = 12
MAX_LEN = 128
VOCAB_SIZE = tokenizer.vocab_size
EMBEDDING_DIM = 32
FEED_FORWARD_DIM = 16
DROPOUT = 0.1
NUM_WORKERS = os.cpu_count()

images = load_dataset("lambdalabs/pokemon-blip-captions")["train"]
length = len(images)
train_size = int(0.8 * length)
val_size = length - train_size

train, val = random_split(images, [train_size, val_size])
images = {"train": train, "test": val}

traindata = PokemonClipDataset(images["train"], context_length=MAX_LEN, tokenizer=tokenizer)
valdata = PokemonClipDataset(images["test"], context_length=MAX_LEN, tokenizer=tokenizer)

traindata = DataLoader(traindata, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
valdata = DataLoader(valdata, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

wandb_logger = WandbLogger(project="clip-pokemon", name="clip-pokemon")
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=100,
)

model = CLIP(NUM_HEADS, NUM_LAYERS, MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM, FEED_FORWARD_DIM, DROPOUT)
trainer.fit(model, traindata, valdata)

