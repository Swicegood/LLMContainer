import requests
import base64
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path):
    base64_image = encode_image(image_path)
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    data = {
        "messages": [
            {
                "role": "system",
                "content": "This is a chat between a user and an assistant. The assistant is helping the user to describe an image.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "model": "llava",
        "max_tokens": 500,
    }

    response = requests.post('http://localhost:8080', headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Test the server
image_path = 'path/to/your/image.jpg'
description = get_image_description(image_path)
print(f"Image description: {description}")