from flask import Flask, request, render_template, send_from_directory
import os 
import urllib.request
import random 
import pathlib 
import openai
import re

def my_image_search_function(folder_path, query):
    # get absolute path to folder_path/ folder given that the folder_path is relative to where the app is running
    folder_path = os.path.join(pathlib.Path(__file__).parent.absolute(), folder_path)
    images = [f for f in os.listdir(folder_path) if query in f]

    return images

def use_llm_to_clipify_user_query(query):
    api_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = api_key
    model_engine = "davinci-003"
    prompt = """
    I am going to give you a query from a user. I want you to extract the relevant keywords and discard anything else, since I am going to be using the output of your query directly with a text encoder. Respond with only the words to query for and nothing else. Here are some example to help you:

    Example 1:
    User: Show me images that have oak trees in the sun
    Response: pictures of oak trees sun

    Example 2:
    User: Which images contain paintings of oranges?
    Response: painting of oranges

    Example 3:
    User: Mustang racing a ferrari. Also I like cats lalalala and my favorite color is blue.
    Response: picture of a mustang racing a ferrari 

    Those are the examples. Now reply to this user query:
    """
    prompt = re.sub(r'[^\w\s]', '', query)
    # Remove excess whitespace
    prompt = re.sub(r'\s+', ' ', query).strip()

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        folder_path = request.form['folder_path']
        images = my_image_search_function(folder_path, query)
        return render_template('results.html', images=images, folder_path=folder_path)
    else:
        return render_template('index.html')

@app.route('/images/<path:filename>/<path:folder>')
def images(filename, folder):
    return send_from_directory(folder, filename)

if __name__ == '__main__':
    app.run(port=5000)