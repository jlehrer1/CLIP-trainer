import argparse
import os

import ray.train
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from ray.air import session
from ray.train.torch import TorchTrainer
from transformers import AutoTokenizer
from ray.air import ScalingConfig
import wandb
from lightning_module import CLIPModel
from pairdataset import PokemonClipDataset
from torch.utils.data import random_split
import ray.data
import ray.train.torch
from torch.utils.data import DataLoader
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air import RunConfig
from ray.air.integrations.wandb import setup_wandb
from torchvision.models import resnet50

model = resnet50()
model.fc = nn.Linear(model.fc.in_features, embedding_dim)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def calculate_loss(
    batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module, text_criterion: nn.Module, image_criterion: nn.Module
):
    image, text = batch
    image_logits, text_logits = model(text, image)
    text_loss = text_criterion(text_logits, torch.arange(text_logits.shape[0]).to(text_logits.device))
    image_loss = image_criterion(image_logits, torch.arange(image_logits.shape[0]).to(image_logits.device))
    loss = (text_loss + image_loss) / 2

    # wandb.log({"text_loss": text_loss, "image_loss": image_loss, "loss": loss})
    return loss


def clip_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    text_criterion: nn.Module,
    image_criterion: nn.Module,
):
    model.train()
    loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        loss = calculate_loss(batch, model, text_criterion, image_criterion)
        loss.backward()
        optimizer.step()

    loss /= len(dataloader)
    return loss


def clip_test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    text_criterion: nn.Module,
    image_criterion: nn.Module,
):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = calculate_loss(batch, model, text_criterion, image_criterion)

    loss /= len(dataloader)
    return loss


def training_loop(config):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    model_params = config["model_params"]
    train, val = config["train"], config["val"]
    setup_wandb()

    batch_size_per_workers = max(1, batch_size // session.get_world_size())

    train_dataset = DataLoader(
        dataset=train,
        batch_size=batch_size_per_workers,
        shuffle=True,
    )
    val_dataset = DataLoader(
        dataset=val,
        batch_size=batch_size_per_workers,
        shuffle=False,
    )
    train_dataset = ray.train.torch.prepare_data_loader(train_dataset)
    train_dataset = ray.train.torch.prepare_data_loader(train_dataset)

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
            dataloader=train_dataset,
            optimizer=optimizer,
            text_criterion=text_criterion,
            image_criterion=image_criterion,
        )
        val_loss = clip_test_epoch(
            model=model,
            dataloader=val_dataset,
            text_criterion=text_criterion,
            image_criterion=image_criterion,
        )
        session.report({"train_loss_epoch": train_loss, "val_loss_epoch": val_loss})
        print(f"Epoch {epoch}: train loss: {train_loss}, val loss: {val_loss}")


def main():
    parser = argparse.ArgumentParser(description="Train CLIP model on Pokemon dataset.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in the multihead attention layer")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the encoder")
    parser.add_argument("--max_len", type=int, default=16, help="Maximum length of the text")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Embedding dimension of the text")
    parser.add_argument("--feed_forward_dim", type=int, default=16, help="Feed forward dimension of the text")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    args = parser.parse_args()
    vocab_size = tokenizer.vocab_size

    images = load_dataset("lambdalabs/pokemon-blip-captions")["train"]
    length = len(images)
    train_size = int(0.8 * length)
    val_size = length - train_size

    train, val = random_split(images, [train_size, val_size])
    images = {"train": train, "test": val}

    train = PokemonClipDataset(images["train"], context_length=args.max_len, tokenizer=tokenizer)
    val = PokemonClipDataset(images["test"], context_length=args.max_len, tokenizer=tokenizer)

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
        "train": train,
        "val": val,
    }
    wandb_callback = WandbLoggerCallback(project="clip-pokemon")
    trainer = TorchTrainer(
        train_loop_per_worker=training_loop,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=args.num_workers),
        run_config=RunConfig(callbacks=[wandb_callback]),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
