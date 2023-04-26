import os 
import pathlib 
from clip_search.image_query.cache_embeddings import get_text_encoding_from_response, find_nearest_filenames, encode_and_pickle_images
from transformers import CLIPModel, CLIPTokenizer
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def get_images(embedding_folder: str, images_folder: str, query: str):
    os.makedirs(embeddings_folder, exist_ok=True)
    print(f"Encoding images in {embedding_folder}...")
    # encode_and_pickle_images(
    #     embeddings_folder=embeddings_folder,
    #     images_folder=images_folder,
    #     clip_model=model,
    #     batch_size=4,
    #     n_embedding_per_file=16,
    #     filenames_txt="filenames.txt"
    # )
    encoding = get_text_encoding_from_response(query=query, tokenizer=tokenizer, clip_model=model)
    # exit(0)
    filenames = find_nearest_filenames(
        embeddings_folder=embeddings_folder,
        query_vector=encoding,
        nlist=2,
        K=10,
        index_file_name="index.faiss",
        filenames_file="filenames.txt"
    )
    print(filenames)
    return filenames

if __name__ == '__main__':
    embeddings_folder = pathlib.Path(__file__).parent.absolute() / "embeddings"
    images_folder = pathlib.Path(__file__).parent.absolute() / pathlib.Path("./clip_search/image_query/images")
    res = get_images(
        embedding_folder=embeddings_folder,
        images_folder=images_folder,
        query="tree",
    )

    print(res)