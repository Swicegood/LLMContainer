# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Pre-set the timezone to avoid prompts
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive


# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
    
# Set the working directory
WORKDIR /app


COPY test.jpg llama-llava-cli ./

RUN chmod +x llama-llava-cli

RUN ls

# Expose the port the server will run on
EXPOSE 8080

# Set the command to run the server
CMD ["/app/llama-llava-cli", \
     "--m", "/models/llava-v1.6-mistral-7b.Q8_0.gguf", \
     "--mmproj", "/models/mmproj-model-f16.gguf", \
     "--image", "test.jpg"]