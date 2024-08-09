# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Pre-set the timezone to avoid prompts
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive


# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libcurl4-openssl-dev \
    git \
    wget \
    curl \
    gdb \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
    
# Set the working directory
WORKDIR /app

# Clone the llama.cpp repository (adjust the URL if you're using a fork)
RUN git clone https://github.com/ggerganov/llama.cpp.git

# Copy your project files
# Copy the source files
COPY CMakeLists.txt llava-server.cpp llama_build_info.cpp config.h.in /app/llama.cpp/examples/llava/

RUN mkdir -p /app/llama.cpp/examples/llava/include/nlohmann

COPY json.hpp /app/llama.cpp/examples/llava/include/nlohmann/


# Build the project
RUN cd llama.cpp/examples/llava && \
    cmake -DCMAKE_C_COMPILER=/usr/bin/gcc \
          -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
          -DLLAVA_DEBUG=ON \
          -DLLAVA_CUDA=ON . && \
    make VERBOSE=1 -j$(nproc)

RUN ls

# Expose the port the server will run on
EXPOSE 8080

# Set the command to run the server
CMD ["/app/llama.cpp/examples/llava/llava-server", \
     "--model", "/models/llava-v1.6-mistral-7b.Q8_0.gguf", \
     "--mmproj", "/models/mmproj-model-f16.gguf", \
     "--port", "8080"]