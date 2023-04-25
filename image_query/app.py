from flask import Flask, request, render_template, send_from_directory
import os 
import urllib.request
import random 
import pathlib 
import openai
import re
from ast import literal_eval

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
    I am going to give you an image query from a user. I want you to extract the relevant keywords and discard anything else. You will respond in a dictionary format with {query_1: <your response>, query_2: <your response> ...}. It is your job to infer how many queries the user wants. If the user asks for multiple things, these are multiple queries. 

    It is extremely important to get this correct. Respond with only the formatted query and nothing else. Here are some example to help you:

    Make sure to infer and format the type of media the user wants. For example, if the user asks for "apple airpods" you should infer {query 1: picture of apple airpods}.

    Example 1:
    User: Show me images that have oak trees in the sun
    Response: {query 1: pictures of oak trees sun}

    Example 2:
    User: Which images contain paintings of oranges?
    Response: {query 1: painting of oranges}

    Example 3:
    User: Which images have red or blue muscle cars?
    Response: {query 1: picture of red muscle car, query 2: picture of blue muscle car}

    Example 4:
    User: Show me apple airpods pros or airpod maxs
    Response: {query 1: picture of apple airpod pro, query 2: picture of apple airpod max}

    Please respond to this query:
    """
    prompt = re.sub(r'[^\w\s]', '', query)
    prompt = re.sub(r'\s+', ' ', query).strip()

    query = re.sub(r'[^\w\s]', '', query)
    query = re.sub(r'\s+', ' ', query).strip()

    prompt = prompt + "\n" + query
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.7,
    )
    response = response.choices[0].text
    try:
        response = literal_eval(response)
    except ValueError:
        raise RuntimeError(f"Model returned incorrectly formatted response: {response}")
        
    return response


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