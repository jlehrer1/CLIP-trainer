import argparse
import os
import boto3
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.loggers import WandbLogger
from modeling.model_definition import CLIP
from torchvision.transforms import Compose, Resize, ToTensor
from modeling.utils import generate_image_text_pair, tokenizer, fetch_images, S3ModelCheckpointer

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
    parser.add_argument("--project", type=str, default="clip-model-training", help="Name of the project for wandb")
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

    with open("credentials", "r") as f:
        credentials = f.read().splitlines()
    s3_client = boto3.client("s3", aws_access_key_id=credentials[0], aws_secret_access_key=credentials[1], endpoint_url="https://s3-west.nrp-nautilus.io/")
    s3_model_checkpointer = S3ModelCheckpointer(
        s3_client=s3_client,
        bucket="braingeneersdev",
        prefix="jlehrer/checkpoints/",
        n_epochs=1,
        n_steps=10000,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[s3_model_checkpointer]
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
