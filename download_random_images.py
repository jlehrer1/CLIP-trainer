import os
import requests
import random
import time
import argparse 

N = 200  # number of images to download
WIDTH = 800  # desired width of images
HEIGHT = 600  # desired height of images
OUTPUT_FOLDER = 'images'  # folder to save images in

parser = argparse.ArgumentParser()
parser.add_argument('--num_images', type=int, default=200, help='Number of images to download')
args = parser.parse_args()
# create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for i in range(args.num_images):
    # generate a random image ID between 1 and 1000
    image_id = random.randint(1, 1000)

    # construct the URL for the image with the desired size
    url = f'https://picsum.photos/id/{image_id}/{WIDTH}/{HEIGHT}'

    # send a GET request to the URL and download the image
    retry_count = 3
    while retry_count > 0:
        try:
            response = requests.get(url)
            response.raise_for_status()  # raise an exception if response is not OK
            filename = os.path.join(OUTPUT_FOLDER, f'image_{image_id}.jpg')
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {filename}')
            break  # exit the loop on success
        except Exception as e:
            print(f'Error downloading image: {e}')
            retry_count -= 1
            if retry_count > 0:
                print(f'Retrying with another image ID...')
                time.sleep(1)  # wait for 1 second before retrying
                image_id = random.randint(1, 1000)
                url = f'https://picsum.photos/id/{image_id}/{WIDTH}/{HEIGHT}'
    else:
        print(f'Failed to download image after 3 retries.')