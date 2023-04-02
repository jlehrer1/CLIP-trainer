from lightning_module import CLIP
from pairdataset import PokemonClipDataset
import pytorch_lightning as pl
import torch 
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import os
import argparse 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

if __name__ == "__main__":
    # take in command line arguments for hyperparameters
    parser = argparse.ArgumentParser(help="Train CLIP model on Pokemon dataset.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in the multihead attention layer")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the encoder")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length of the text")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dimension of the text")
    parser.add_argument("--feed_forward_dim", type=int, default=16, help="Feed forward dimension of the text")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for dataloader")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")

    args = parser.parse_args()
    VOCAB_SIZE = tokenizer.vocab_size

    images = load_dataset("lambdalabs/pokemon-blip-captions")["train"]
    length = len(images)
    train_size = int(0.8 * length)
    val_size = length - train_size

    train, val = random_split(images, [train_size, val_size])
    images = {"train": train, "test": val}

    traindata = PokemonClipDataset(images["train"], context_length=args.max_len, tokenizer=tokenizer)
    valdata = PokemonClipDataset(images["test"], context_length=args.max_len, tokenizer=tokenizer)

    traindata = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valdata = DataLoader(valdata, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    wandb_logger = WandbLogger(project="clip-pokemon", name="clip-pokemon")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=100,
    )

    model = CLIP(
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        dropout=args.dropout,
        vocab_size=VOCAB_SIZE,
    )

    trainer.fit(model, traindata, valdata)

