import os 
import pathlib 
from clip_search.image_query.cache_embeddings import get_text_encoding_from_response, find_nearest_filenames, encode_and_pickle_images

def get_images(folder_path: str, query: str):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    embeddings_folder = pathlib.Path(__file__).parent.absolute() / "embeddings"
    os.makedirs(embeddings_folder, exist_ok=True)
    print(f"Encoding images in {folder_path}...")
    encode_and_pickle_images(
        embeddings_folder=embeddings_folder,
        images_folder="./images",
        clip_model=model,
        batch_size=4,
        n_embedding_per_file=100,
        filenames_txt="filenames.txt"
    )
    exit(0)
    encoding = get_text_encoding_from_response(query=query, tokenizer=tokenizer, clip_model=model)
    filesnames = find_nearest_filenames(
        embeddings_folder=embeddings_folder,
        query_vector=encoding,
        nlist=100,
        K=10,
        index_file_name="index.faiss",
        filenames_file="filenames.txt"
    )

    return filesnames

if __name__ == '__main__':
    res = get_images(
        "./images",
        "tree",
    )

    print(res)