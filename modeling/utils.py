import torch
from transformers import AutoTokenizer
from typing import Any
from torchvision.transforms import Compose
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import pytorch_lightning as pl
import PIL.Image
import os 
import boto3
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


class S3ModelCheckpointer(pl.Callback):
    def __init__(self, s3_client: boto3.client, bucket: str, prefix: str, n_epochs: int = 1, n_steps: int = 1000):
        """
        A callback to save checkpoints to S3.

        :param s3_client: The S3 client to use.
        :param bucket: The bucket to save the checkpoints to.
        :param prefix: The prefix to save the checkpoints to.
        :param n_epochs: The number of epochs between checkpoints.
        :param n_steps: The number of steps between checkpoints.
        """
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix
        self.n_epochs = n_epochs
        self.n_steps = n_steps

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs) -> None:
        print(f"Uploading model checkpoint on step {trainer.global_step}.")
        if trainer.global_step % self.n_steps == 0:
            self._save_checkpoint(trainer)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print(f"Uploading model checkpoint on epoch {trainer.current_epoch}.")
        if trainer.current_epoch % self.n_epochs == 0:
            self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer: pl.Trainer, epoch: int, step: int) -> None:
        trainer.save_checkpoint(f"checkpoint-{trainer.current_epoch}-{trainer.global_step}.ckpt")
        self.s3_client.upload_file(
            f"checkpoint-{trainer.global_step}.ckpt", 
            self.bucket, 
            os.path.join(self.prefix, f"checkpoint-{trainer.current_epoch}-{trainer.global_step}.ckpt")
        )