from openai import OpenAI

client = OpenAI()

import base64
import requests
import secret
import json

# OpenAI API Key
api_key = secret.OPENAI_API_KEY


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "adjusted_critical_images/Screenshot 2024-04-26 151032.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "I am going to give you a sequence of images one image at a time that represents the frames of a change in direction from a video of the route. Based off of the provided image, please give a concise description of the area as if you were explaining the route to a visually impaired person. Please avoid mentioning the sense of smell and taste. You may assume that the visually impaired person traversing this route is adept with a white cane.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
)
print(response.json())

# Add to the json file
JSON_PATH = "Hal_to_206_description.json"
previous_response = json.loads(JSON_PATH)
updated_response = previous_response.update(response.json())

with open(JSON_PATH, "w") as outfile:
    json.dump(updated_response, outfile)
