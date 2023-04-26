import os 
import torch 
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import faiss
import pickle
import openai
import re
from ast import literal_eval
from transformers import CLIPImageProcessor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.image_paths = sorted(os.listdir(folder_path))
        self.image_preprocesser = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.image_preprocesser(images=image, return_tensors="pt")
        image["pixel_values"] = image["pixel_values"].squeeze(0)
        return image, self.image_paths[idx]

def encode_and_pickle_images(embeddings_folder: str, images_folder: str, clip_model: torch.nn.Module, batch_size: int = 16, n_embedding_per_file: int = 128, filenames_txt: str = "filenames.txt"):
    """
    Encodes all images in a folder using CLIP and saves the embeddings to .npy files.

    :param embeddings_folder: The folder containing the images to encode.
    :param clip_model: The CLIP model to use for encoding.
    :param batch_size: The batch size to use for encoding.
    :param n_embedding_per_file: The number of embeddings to save per .npy file.
    :param filenames_txt: The name of the .txt file to save the filenames to.
    """
    assert batch_size < n_embedding_per_file, "batch_size must be smaller than n_embedding_per_file"
    assert n_embedding_per_file % batch_size == 0, "n_embedding_per_file must be a multiple of batch_size"
    os.makedirs(embeddings_folder, exist_ok=True)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set model to evaluation mode
    clip_model.eval()
    # Create dataloader
    dataset = ImageFolderDataset(images_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Get embeddings shape
    sample = dataset[0][0]
    sample["pixel_values"] = sample["pixel_values"].unsqueeze(0)
    embeddings_shape = clip_model.get_image_features(**sample).shape[1]
    # Loop over images in batches and save embeddings and filenames to .npy files
    embeddings_batch = np.empty((0, embeddings_shape))
    filenames_batch = []
    file_index = 0
    with torch.no_grad():
        for i, (batch, filenames) in enumerate(dataloader):
            # check if numpy file exists, if so we continue to the next iter 
            if os.path.exists(os.path.join(embeddings_folder, f"{file_index}.npy")):
                file_index += 1
                continue

            batch = batch.to(device)
            embeddings = clip_model.get_image_features(**batch).cpu().numpy()
            embeddings_batch = np.concatenate((embeddings_batch, embeddings), axis=0)
            filenames_batch.extend(filenames)
            if len(embeddings_batch) == n_embedding_per_file:
                embeddings_file_name = os.path.join(embeddings_folder, f"{file_index}.npy")
                np.save(embeddings_file_name, embeddings_batch.astype(np.float32))
                print(f"Saved embeddings up to batch {i}, continuing...")
                with open(os.path.join(embeddings_folder, filenames_txt), "a") as f:
                    for filenames in filenames_batch:
                        f.write(filenames + "\n")
                filenames_batch = []
                embeddings_batch = np.empty((0, embeddings_shape))
                file_index += 1
        # Save last .npy file of embeddings
        if len(embeddings_batch) > 0:
            embeddings_file_name = os.path.join(embeddings_folder, f"{file_index + 1}.npy")
            np.save(embeddings_file_name, embeddings_batch.astype(np.float32))
            print(f"Saved embeddings for final batches")
            with open(os.path.join(embeddings_folder, filenames_txt), "a") as f:
                for filenames in filenames_batch:
                    f.write(filenames + "\n")

def find_nearest_filenames(embeddings_folder: str, query_vector: np.ndarray, nlist: int, K: int, index_file_name: str, filenames_file = "filenames.txt"):
    """
    Uses FAISS to find the filenames of the images with the nearest embeddings to the query vector.

    :param embeddings_folder: The folder containing the embeddings.
    :param query_vector: The query vector.
    :param nlist: The number of cells to use for the IVF index.
    :param K: The number of nearest neighbors to return.
    :param index_file_name: The name of the pickled index file in embeddings_folder.
    :param filenames_file: The name of the .txt file containing the filenames. Defaults to "filenames.txt".

    :return: A list of the filenames of the images with the nearest embeddings to the query vector.
    """

    # Set embedding size based on first file in folder
    for file_name in os.listdir(embeddings_folder):
        if file_name.endswith(".npy"):
            batch_size, embedding_size = np.load(os.path.join(embeddings_folder, file_name), allow_pickle=True).shape
            print(f"Embedding size: {embedding_size}")
            print(f"Batch size: {batch_size}")
            break
    else:
        raise ValueError("No .npy files found in the embeddings folder.")

    # assert batch_size < nlist, "batch_size must be less than the number of cells in the IVF index"
    # Check if pickled index exists
    index_file_name = os.path.join(embeddings_folder, index_file_name)
    filenames_file = os.path.join(embeddings_folder, filenames_file)
    if os.path.exists(index_file_name):
        # Load pickled index
        with open(index_file_name, "rb") as f:
            index = pickle.load(f)
    else:
        # Create new index
        coarse_quantizer = faiss.IndexFlatIP(embedding_size)
        index = faiss.IndexIVFFlat(coarse_quantizer, embedding_size, nlist, faiss.METRIC_INNER_PRODUCT)
        embedding_files = [f for f in os.listdir(embeddings_folder) if f.endswith(".npy")]
        for file_name in embedding_files:
            embedding = np.load(os.path.join(embeddings_folder, file_name), allow_pickle=True).astype(np.float32)
            # L2 normalization of embedding vectors since we are using cosine similarity
            embedding /= np.linalg.norm(embedding, axis=1)[:, np.newaxis]
            index.train(embedding)
            index.add(embedding)
        # Save index to disk
        with open(index_file_name, "wb") as f:
            pickle.dump(index, f)
    # Search for nearest neighbors
    query_vector = query_vector.astype(np.float32)
    print(f"Query vector shape: {query_vector.shape}")
    _, indices = index.search(query_vector, K)
    # Read filenames from file
    with open(filenames_file, "r") as f:
        filenames = [line.strip() for line in f.readlines()]
    # Get filenames at indices
    return [filenames[i] for i in indices[0]]

def get_text_encoding_from_response(query: str, tokenizer: nn.Module, clip_model: nn.Module) -> torch.Tensor:
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        raise KeyError("Please set the OPENAI_API_KEY environment variable to your OpenAI API key.")

    openai.api_key = api_key
    model_engine = "gpt-3.5-turbo"
    prompt = """
    It is extremely important to get this correct. I am going to give you an image query from a user. I want you to extract the relevant keywords and discard anything else. You will respond in a Python list format like [response 1, response 2, ...] where each response is a string.
    It is your job to infer how many queries the user wants. If the user asks for multiple things, these are multiple queries.

    Respond with only the formatted query and nothing else. Here are some example to help you:

    Make sure to infer and format the type of media the user wants. For example, if the user asks for "apple airpods" you should infer ["picture of apple airpods"].

    Example 1:
    User: Show me images that have oak trees in the sun
    Response: ["pictures of oak trees sun"]

    Example 2:
    User: Which images contain paintings of oranges?
    Response: ["painting of oranges"]

    Example 3:
    User: Which images have red or blue muscle cars?
    Response: ["picture of red muscle car", "picture of blue muscle car"]

    Example 4:
    User: Show me apple airpods pros or airpod maxs
    Response: ["picture of apple airpod pro", "picture of apple airpod max"]

    Please respond to this query:
    """
    prompt = prompt + "\n" + query
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.5,
    )
    response = response.choices[0].message.content
    # Yes I know this is bad but it's only running locally so if you prompt inject 
    # some stuff and break your computer 
    # with malicious python code it's kinda your fault
    try:
        response = literal_eval(response)
    except:
        raise ValueError(f"Model returned an invalid response {response}. Please try again.")
    encoded_text = tokenizer.encode(response, return_tensors="pt")
    encoded_text = clip_model.get_text_features(encoded_text)
    return encoded_text.detach().cpu().numpy()