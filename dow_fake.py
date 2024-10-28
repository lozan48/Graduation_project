import requests
from PIL import Image
from io import BytesIO
import os

def download_generated_faces(directory, num_images):
    os.makedirs(directory, exist_ok=True)
    url = "https://thispersondoesnotexist.com/"
    for i in range(num_images):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(directory, f"face_{i}.jpg"))
        if i % 10 == 0:
            print(f"Downloaded {i+1}/{num_images} images")


# Example usage
download_generated_faces('ai_generated_faces', 1)