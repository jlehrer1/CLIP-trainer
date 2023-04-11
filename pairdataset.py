import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms import Compose, Resize, ToTensor
from transformers import AutoTokenizer
import requests
from PIL import Image


class BaseClipDataset(Dataset):
    def __init__(self, dataset, context_length: int, image_transform=None, tokenizer=None, image_key="image", text_key="text"):
        self.dataset = dataset
        self.context_length = context_length
        self.image_transform = Compose([Resize((224, 224)), ToTensor()]) if image_transform is None else image_transform
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") if tokenizer is None else tokenizer
        self.image_key = image_key
        self.text_key = text_key

    def read_image(self, url: str):
        try:  # Try to read the image
            image = Image.open(requests.get(url, stream=True).raw)
        except:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        return image

    def process_sample(self, image, text):
        if isinstance(image, str):
            image = self.read_image(image)
        if self.image_transform is not None:
            image = self.image_transform(image)
        text = self.tokenizer.encode(
            text, add_special_tokens=True, max_length=self.context_length, truncation=True, padding="max_length"
        )
        text = torch.tensor(text, dtype=torch.long)
        return image, text


class ClipDataset(BaseClipDataset):
    def __init__(self, dataset, context_length: int, image_transform=None, tokenizer=None, image_key="image", text_key="text"):
        super().__init__(dataset, context_length, image_transform, tokenizer, image_key, text_key)

    def __getitem__(self, idx):
        image, text = self.dataset[idx][self.image_key], self.dataset[idx][self.text_key]
        return self.process_sample(image, text)

    def __len__(self):
        return len(self.dataset)


class IterableClipDataset(BaseClipDataset, IterableDataset):
    def __init__(self, dataset, context_length: int, image_transform=None, tokenizer=None, image_key="image", text_key="text"):
        super().__init__(dataset, context_length, image_transform, tokenizer, image_key, text_key)

    def __iter__(self):
        for sample in self.dataset:
            image, text = sample[self.image_key], sample[self.text_key]
            yield self.process_sample(image, text)


class IterableClipDataset(IterableDataset):
    def __init__(self, dataset, context_length: int, image_transform=None, tokenizer=None, image_key="image", text_key="text"):
        self.dataset = dataset
        self.context_length = context_length
        self.image_transform = Compose([Resize((224, 224)), ToTensor()]) if image_transform is None else image_transform
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") if tokenizer is None else tokenizer
        self.image_key = image_key
        self.text_key = text_key

    def read_image(self, url: str):
        try:  # Try to read the image
            image = Image.open(requests.get(url, stream=True).raw)
        except:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        return image

    def process_sample(self, image, text):
        if isinstance(image, str):
            image = self.read_image(image)
        if self.image_transform is not None:
            image = self.image_transform(image)
        text = self.tokenizer.encode(
            text, add_special_tokens=True, max_length=self.context_length, truncation=True, padding="max_length"
        )
        text = torch.tensor(text, dtype=torch.long)
        return image, text

    def __iter__(self):
        for sample in self.dataset:
            print("sample in dataset is", sample)
            image, text = sample[self.image_key], sample[self.text_key]
            yield self.process_sample(image, text)
