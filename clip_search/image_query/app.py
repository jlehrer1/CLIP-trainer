import os
import pathlib

from flask import Flask, render_template, request, send_from_directory
from transformers import CLIPModel, CLIPProcessor

from clip_search.image_query.cache_embeddings import (
    encode_and_pickle_images,
    find_nearest_filenames,
    get_text_encoding_from_response,
)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
full_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = full_processor.tokenizer
image_processor = full_processor.image_processor


def get_images(embeddings_folder: str, images_folder: str, query: str):
    encode_and_pickle_images(
        embeddings_folder=embeddings_folder,
        images_folder=images_folder,
        clip_model=model,
        image_preprocessor=image_processor,
        batch_size=4,
        n_embedding_per_file=16,
        filenames_txt="filenames.txt",
    )
    encoding = get_text_encoding_from_response(query=query, tokenizer=tokenizer, clip_model=model)
    filenames = find_nearest_filenames(
        embeddings_folder=embeddings_folder,
        query_vector=encoding,
        nlist=2,
        K=3,
        index_file_name="index.faiss",
        filenames_file="filenames.txt",
    )
    return filenames


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        folder_path = request.form["folder_path"]
        embeddings_folder = pathlib.Path(__file__).parent.absolute() / "embeddings"
        images_folder = pathlib.Path(__file__).parent.absolute() / pathlib.Path(folder_path)
        images = get_images(embeddings_folder=embeddings_folder, images_folder=images_folder, query=query)
        print("images are", images)
        return render_template("results.html", images=images, folder_path=folder_path)
    else:
        return render_template("index.html")


@app.route("/images/<path:filename>/<path:folder>")
def images(filename, folder):
    return send_from_directory(folder, filename)


if __name__ == "__main__":
    app.run(port=5000)
