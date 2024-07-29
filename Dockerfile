# Use an official C++ base image
FROM gcc:latest

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    libopencv-dev \
    libcurl4-openssl-dev \
    rapidjson-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the source files
COPY . .

# Build the project
RUN mkdir build && cd build && \
    cmake .. && \
    make

# Expose the port the server will run on
EXPOSE 8080

# Set the command to run the server
CMD ["/app/build/llava-server", \
     "--model", "/models/llava-v1.6-mistral-7b.Q8_0.gguf", \
     "--mmproj", "/models/mmproj-model-f16.gguf", \
     "--port", "8080"]