import argparse
import os

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from typing import Any
from model_definition import CLIP
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import requests
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

import PIL.Image

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
USER_AGENT = get_datasets_user_agent()


def generate_image_text_pair(data: dict[str, Any], transforms: Compose, max_length: int = 24):
    """
    Generate a pair of tensor image and tokenized text from a url and text description of the url.

    :param data: The dictionary of data to generate the pair from.
    :param transforms: The transforms to apply to the image.
    :param max_length: The maximum length of the tokenized text.
    """
    images = []
    texts = []
    for text, image in zip(data["text"], data["image"]):
        if image is not None and text is not None:
            image = image.convert("RGB")
            image = transforms(image)
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            text = tokenizer.encode(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

            images.append(image)
            texts.append(text)

    images = torch.stack(images)
    texts = torch.cat(texts, dim=0)
    return {"images": images, "texts": texts}


def fetch_single_image(image_data, timeout=None, retries=0):
    """
    Download a single image from a URL.

    :param image_data: The URL of the image.
    :param timeout: The timeout for the request.
    :param retries: The number of retries to attempt if the request fails.
    """
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_data,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=None, retries=0):
    """
    Download images in parallel using a thread pool.

    :param batch: A batch of data containing the image urls.
    :param num_threads: The number of threads to use for downloading.
    :param timeout: The timeout for the requests.
    :param retries: The number of retries to attempt if a request fails.
    """
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["url"]))
    return batch


if __name__ == "__main__":
    # take in command line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Train CLIP model on Pokemon dataset.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in the multihead attention layer")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the encoder")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length of the text")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension of the text")
    parser.add_argument("--feed_forward_dim", type=int, default=16, help="Feed forward dimension of the text")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for dataloader")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("--project", type=str, default="clip-pokemon", help="Name of the project for wandb")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="kakaobrain/coyo-700m",
        help="Name of the dataset to use from the huggingface Dataset hub",
    )
    args = parser.parse_args()
    VOCAB_SIZE = tokenizer.vocab_size

    train_len = 5000
    val_len = 1000
    remove_columns = [
        "id",
        "width",
        "height",
        "image_phash",
        "text_length",
        "word_count",
        "num_tokens_bert",
        "num_tokens_gpt",
        "num_faces",
        "clip_similarity_vitb32",
        "clip_similarity_vitl14",
        "nsfw_score_opennsfw2",
        "nsfw_score_gantman",
        "watermark_score",
        "aesthetic_score_laion_v2",
        "url",
        "text",
        "image",
    ]
    data = load_dataset("kakaobrain/coyo-700m", split="train", streaming=True, batch_size=args.batch_size)
    train, test = data.take(train_len), data.skip(train_len).take(val_len)

    train = train.map(lambda x: fetch_images(x, num_threads=4), batched=True, batch_size=args.batch_size)
    train = train.map(
        lambda x: generate_image_text_pair(x, transforms=Compose([Resize((224, 224)), ToTensor()]), max_length=args.max_len),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=remove_columns,
    )
    test = test.map(lambda x: fetch_images(x, num_threads=4), batched=True, batch_size=args.batch_size)
    test = test.map(
        lambda x: generate_image_text_pair(x, transforms=Compose([Resize((224, 224)), ToTensor()]), max_length=args.max_len),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=remove_columns,
    )
    train = train.iter(batch_size=args.batch_size)
    test = test.iter(batch_size=args.batch_size)

    wandb_logger = WandbLogger(project=args.project, name=args.project + "-run", log_model=True, save_dir="logs")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.epochs,
        logger=wandb_logger,
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

    trainer.fit(model, train, test)
