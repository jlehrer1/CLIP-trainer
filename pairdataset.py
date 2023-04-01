from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch 

class PokemonClipDataset(Dataset):
    def __init__(self, dataset, context_length: int, image_transform=None, tokenizer=None):
        self.dataset = dataset
        self.context_length = context_length
        self.image_transform = Compose([Resize((224, 224)), ToTensor()]) if image_transform is None else image_transform
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") if tokenizer is None else tokenizer

    def __getitem__(self, idx):
        image, text = self.dataset[idx]['image'], self.dataset[idx]['text']
        if self.image_transform is not None:
            image = self.image_transform(image)

        text = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.context_length, truncation=True, padding="max_length")
        text = torch.tensor(text, dtype=torch.long)
        return image, text

    def __len__(self):
        return len(self.dataset)
