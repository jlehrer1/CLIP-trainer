import argparse
import os

import numpy as np
import pytorch_lightning as pl
import ray.train
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ray.data.preprocessors import Concatenator
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

import wandb
from lightning_module import CLIPModel
from pairdataset import PokemonClipDataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def calculate_loss(batch: tuple, model: nn.Module, text_criterion: nn.Module, image_criterion: nn.Module):
    image, text = batch
    text_logits, image_logits = model(image, text)
    text_loss = text_criterion(text_logits, torch.arange(text_logits.shape[0]).to(text_logits.device))
    image_loss = image_criterion(image_logits, torch.arange(image_logits.shape[0]).to(image_logits.device))
    loss = (text_loss + image_loss) / 2

    wandb.log({"text_loss": text_loss, "image_loss": image_loss, "loss": loss})
    return loss


def clip_training_epoch(
    model: nn.Module,
    dataset: ray.data.Dataset,
    optimizer: optim.Optimizer,
    text_criterion: nn.Module,
    image_criterion: nn.Module,
    batch_size_per_workers: int,
):
    model.train()
    loss = 0
    for batch in dataset.iter_torch_batches(batch_size=batch_size_per_workers):
        optimizer.zero_grad()
        loss = calculate_loss(batch, model, text_criterion, image_criterion)
        loss.backward()
        optimizer.step()

    loss /= len(dataset)
    return loss


def clip_test_epoch(
    model: nn.Module,
    dataset: ray.data.Dataset,
    text_criterion: nn.Module,
    image_criterion: nn.Module,
    batch_size_per_workers: int,
):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in dataset.iter_torch_batches(batch_size=batch_size_per_workers):
            loss = calculate_loss(batch, model, text_criterion, image_criterion)

    loss /= len(dataset)
    return loss


def training_loop(config):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    model_params = config["model_params"]

    train_dataset = session.get_dataset_shard("train")
    val_dataset = session.get_dataset_shard("val")

    batch_size_per_workers = batch_size // session.get_world_size()

    model = CLIPModel(
        num_heads=model_params["num_heads"],
        num_layers=model_params["num_layers"],
        max_len=model_params["max_len"],
        vocab_size=model_params["vocab_size"],
        embedding_dim=model_params["embedding_dim"],
        feed_forward_dim=model_params["feed_forward_dim"],
        dropout=model_params["dropout"],
    )

    model = ray.train.torch.prepare_model(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    text_criterion = nn.CrossEntropyLoss()
    image_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = clip_training_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            text_criterion=text_criterion,
            image_criterion=image_criterion,
            batch_size_per_workers=batch_size_per_workers,
        )
        val_loss = clip_test_epoch(
            model=model,
            dataset=val_dataset,
            text_criterion=text_criterion,
            image_criterion=image_criterion,
            batch_size_per_workers=batch_size_per_workers,
        )
        session.report({"train_loss_epoch": train_loss, "val_loss_epoch": val_loss})
        print(f"Epoch {epoch}: train loss: {train_loss}, val loss: {val_loss}")


def main():
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
    vocab_size = tokenizer.vocab_size

    images = load_dataset("lambdalabs/pokemon-blip-captions")["train"]
    length = len(images)
    train_size = int(0.8 * length)
    val_size = length - train_size

    train, val = random_split(images, [train_size, val_size])
    images = {"train": train, "test": val}

    traindata = PokemonClipDataset(images["train"], context_length=args.max_len, tokenizer=tokenizer)
    valdata = PokemonClipDataset(images["test"], context_length=args.max_len, tokenizer=tokenizer)

    traindata = ray.data.from_torch(traindata)
    valdata = ray.data.from_torch(valdata)

    config = {
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "model_params": {
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "max_len": args.max_len,
            "vocab_size": vocab_size,
            "embedding_dim": args.embedding_dim,
            "feed_forward_dim": args.feed_forward_dim,
            "dropout": args.dropout,
        },
        "train_dataset": traindata,
        "val_dataset": valdata,
    }


if __name__ == "__main__":
    main()
