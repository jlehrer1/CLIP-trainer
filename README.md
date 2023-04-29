# CLIP Search

Web-based image search engine that uses OpenAI's CLIP Model to find and display visually similar images to a user's query text. Capabilities to train your own CLIP model using a HuggingFace dataset, with a ResNet image encoder and a encoder-only transformer for the text encoder.

The CLIP Model, short for Contrastive Language-Image Pre-Training, is a state-of-the-art model that encodes both image and text inputs in a shared embedding space.

Built with Flask, a Python web framework, and uses the CLIP Model, a transformer-based neural network for image and text retrieval developed by OpenAI. The system allows users to input text queries and retrieves the most visually similar images.

To run the code:
Clone the repository with git clone https://github.com/jlehrer1/CLIP-trainer.git
Navigate to the project directory with cd clip-search.
Install required packages with pip install -r requirements.txt.
Start the Flask web server with python app.py.
Navigate to http://localhost:5000 in a web browser.