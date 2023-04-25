from flask import Flask, request, render_template, send_from_directory
import os 
import pathlib 
from image_query.cache_embeddings import get_text_encoding_from_response, find_nearest_filenames, encode_and_pickle_images

def get_images(folder_path: str, query: str):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    embeddings_folder = pathlib.Path(__file__).parent.absolute() / "embeddings"
    os.makedirs(embeddings_folder, exist_ok=True)
    print(f"Encoding images in {folder_path}...")
    encode_and_pickle_images(
        embeddings_folder=embeddings_folder,
        folder_path=folder_path,
        model=model,
        batch_size=4,
        n_embedding_per_file=100,
        filenames_txt="filenames.txt"
    )

    encoding = get_text_encoding_from_response(tokenizer, query)
    filesnames = find_nearest_filenames(
        embeddings_folder=embeddings_folder,
        query_vector=encoding,
        nlist=100,
        K=10,
        index_file_name="index.faiss"
        filenames_file="filenames.txt"
    )

    return filesnames

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        folder_path = request.form['folder_path']

        images = get_images(folder_path, query)
        return render_template('results.html', images=images, folder_path=folder_path)
    else:
        return render_template('index.html')

@app.route('/images/<path:filename>/<path:folder>')
def images(filename, folder):
    return send_from_directory(folder, filename)

if __name__ == '__main__':
    app.run(port=5000)