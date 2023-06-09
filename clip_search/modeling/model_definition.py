import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from clip_search.modeling.resnet_encoder import ResNetX
from clip_search.modeling.text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, num_heads, num_layers, max_len, vocab_size, embedding_dim, feed_forward_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.feed_forward_dim = feed_forward_dim
        self.dropout = dropout

        # we use a transformer encoder for the text
        # which projects a batch of shape (batch_size, max_length) into a vector of shape (batch_size, embedding_dim)
        # the embedding_dim is the same as the image embedding so we can calculate the similarity between the two
        self.text_encoder = TextEncoder(
            self.num_heads,
            self.num_layers,
            self.max_len,
            self.vocab_size,
            self.embedding_dim,
            self.feed_forward_dim,
            self.dropout,
        )

        # we use a resnet50 to encode the image and set the final fully connected layer to have the same embedding dimension
        # as the text encoder
        self.image_encoder = resnet50()
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, self.embedding_dim)

        self.text_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.embedding_dim),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.embedding_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text = self.text_encoder(text)
        image = self.image_encoder(image)

        image_features = image / image.norm(dim=-1, keepdim=True)
        text_features = text / text.norm(dim=-1, keepdim=True)
        # this serves as a temperature parameter
        logit_scale = self.logit_scale.exp()
        # we calculate the logits by taking the dot product between the image and text features
        # this achieves the same thing as the cosine similarity
        # we also scale the logits by a factor of 1 / temperature

        # text_features: (batch_size, embedding_dim)
        # image_features: (batch_size, embedding_dim)
        # logits_per_text: (batch_size, batch_size)
        # and the same for logits_per_image
        logits_per_text = logit_scale * text_features @ image_features.t()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return logits_per_image, logits_per_text


class CLIP(pl.LightningModule):
    def __init__(self, num_heads, num_layers, max_len, vocab_size, embedding_dim, feed_forward_dim, dropout):
        """
        Defines the CLIP model architecture, a model that takes in an image and text and outputs the similarity between the two.
        It uses a transformer encoder for the text and a resnet50 for the image encoder. For a batch of N samples,
        the model is then trained  to maximize similarity between matching samples of (text, image) data
        while minimizing similarity between other the N^2 - N pairs of (text, image) data.

        :ivar: num_heads: the number of heads in the multi-head attention layer
        :ivar: num_layers: the number of layers in the transformer encoder
        :ivar: max_len: the maximum length of the text, also known as the context_length in the transformer encoder
        :ivar: vocab_size: the size of the tokenizer vocabulary
        :ivar: embedding_dim: the dimension of the embedding layer, which such that batches will have shape (batch_size, embedding_dim)
        :ivar: feed_forward_dim: the dimension of the feed forward layer in the transformer encoder
        :ivar: dropout: the dropout rate
        """
        super().__init__()
        self.model = CLIPModel(num_heads, num_layers, max_len, vocab_size, embedding_dim, feed_forward_dim, dropout)
        self.image_loss = torch.nn.CrossEntropyLoss()
        self.text_loss = torch.nn.CrossEntropyLoss()

    def __step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        image, text = batch
        logits_per_image, logits_per_text = self.model(text, image)

        # logits_per_text: (batch_size, batch_size)
        # The diagonal elements are the logits for the image being the same as the text
        # so we want to maximize the diagonal elements which is why we use
        # a list of labels from 0 to batch_size - 1.

        # so for the ith image, we want the ith text to be the same as the image
        # otherwise, we want the text to be different from the image
        # So the labels for the ith image are [0, 1, 2, ..., i - 1, i + 1, ..., batch_size - 1]
        image_loss = self.image_loss(logits_per_image, torch.arange(logits_per_image.shape[0]).to(logits_per_image.device))
        text_loss = self.text_loss(logits_per_text, torch.arange(logits_per_text.shape[0]).to(logits_per_text.device))

        loss = (image_loss + text_loss) / 2
        return loss

    def training_step(self, batch, batch_idx):
        images, texts = batch["images"], batch["texts"]
        if isinstance(images, list):
            images = torch.stack(images)

        if isinstance(texts, list):
            texts = torch.stack(texts)

        loss = self.__step((images, texts))
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, texts = batch["images"], batch["texts"]
        if isinstance(images, list):
            images = torch.stack(images)

        if isinstance(texts, list):
            texts = torch.stack(texts)

        loss = self.__step((images, texts))
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [lr_scheduler]
