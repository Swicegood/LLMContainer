# LLaVA Server

This project implements a server with an OpenAI-compatible API for image-to-text generation using the LLaVA (Large Language and Vision Assistant) model. It provides a dockerized environment for easy deployment and use.

## Features

- OpenAI-compatible API for image-to-text generation
- Dockerized environment for easy setup and deployment
- Support for LLaVA 1.6 with Mistral 7B model
- Efficient handling of large model files using Docker volumes

## Prerequisites

- Docker
- curl (for downloading model files)
- Python 3.7+ (for running the test script)

## Quick Start

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llava-server.git
   cd llava-server
   ```

2. Build the Docker image:
   ```
   docker build -t llava-server .
   ```

3. Create a directory for the model files:
   ```
   mkdir -p /path/to/model/directory
   ```

4. Download the model files:
   ```
   curl -L https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q8_0.gguf -o /path/to/model/directory/llava-v1.6-mistral-7b.Q8_0.gguf
   curl -L https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf -o /path/to/model/directory/mmproj-model-f16.gguf
   ```

5. Run the Docker container:
   ```
   docker run -p 8080:8080 -v /path/to/model/directory:/models llava-server
   ```

Alternatively, you can use the provided setup script:

1. Edit the `MODEL_DIR` variable in `run_llava_server.sh` to point to your desired model directory.
2. Make the script executable:
   ```
   chmod +x run_llava_server.sh
   ```
3. Run the script:
   ```
   ./run_llava_server.sh
   ```

## Usage

Once the server is running, you can send requests to `http://localhost:8080` using the following format:

```python
import requests
import base64

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

# Example usage
image_path = 'path/to/your/image.jpg'
description = get_image_description(image_path)
print(f"Image description: {description}")
```

## Project Structure

- `Dockerfile`: Defines the Docker image for the server
- `llava-server.cpp`: Main server implementation
- `CMakeLists.txt`: CMake configuration for building the server
- `run_llava_server.sh`: Script to automate model download and server startup
- `test_script.py`: Python script to test the server

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the LLaVA model developed by Microsoft Research
- Thanks to the creators of llama.cpp for their efficient C++ implementation