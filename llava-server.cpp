#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include "clip.h"
#include "llama.h"
#include "llava.h"

// Function declarations
void handle_client(int client_socket);
std::string process_request(const std::string& request_body);
std::string generate_image_description(const std::string& image_data);
std::string base64_decode(const std::string& encoded_string);

// Global variables
struct clip_ctx* clip_ctx;
llama_model* llama_model;
llama_context* llama_ctx;

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string model_path, mmproj_path;
    int port = 8080;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::string(argv[i]) == "--mmproj" && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (std::string(argv[i]) == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        }
    }

    if (model_path.empty() || mmproj_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --model <path> --mmproj <path> [--port <port>]" << std::endl;
        return 1;
    }

    // Initialize CLIP
    clip_ctx = clip_model_load(mmproj_path.c_str(), 1);
    if (!clip_ctx) {
        std::cerr << "Failed to load CLIP model" << std::endl;
        return 1;
    }

    // Initialize LLaMA
    llama_backend_init(false);
    llama_model_params model_params = llama_model_default_params();
    llama_model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!llama_model) {
        std::cerr << "Failed to load LLaMA model" << std::endl;
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    llama_ctx = llama_new_context_with_model(llama_model, ctx_params);
    if (!llama_ctx) {
        std::cerr << "Failed to create LLaMA context" << std::endl;
        return 1;
    }

    // Set up server socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Failed to bind to port " << port << std::endl;
        return 1;
    }

    if (listen(server_fd, 3) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
        return 1;
    }

    std::cout << "Server listening on port " << port << std::endl;

    while (true) {
        int client_socket = accept(server_fd, nullptr, nullptr);
        if (client_socket < 0) {
            std::cerr << "Failed to accept client connection" << std::endl;
            continue;
        }

        std::thread(handle_client, client_socket).detach();
    }

    return 0;
}

void handle_client(int client_socket) {
    char buffer[1024] = {0};
    std::string request;

    while (true) {
        int valread = read(client_socket, buffer, 1024);
        if (valread <= 0) break;
        request.append(buffer, valread);
        if (request.find("\r\n\r\n") != std::string::npos) break;
    }

    std::string response = process_request(request);

    send(client_socket, response.c_str(), response.length(), 0);
    close(client_socket);
}

std::string process_request(const std::string& request_body) {
    rapidjson::Document d;
    d.Parse(request_body.c_str());

    if (d.HasParseError() || !d.IsObject()) {
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Invalid JSON\"}";
    }

    if (!d.HasMember("image") || !d["image"].IsString()) {
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Missing or invalid 'image' field\"}";
    }

    std::string image_data = d["image"].GetString();
    std::string description = generate_image_description(image_data);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("description");
    writer.String(description.c_str());
    writer.EndObject();

    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";
    response += buffer.GetString();

    return response;
}

std::string generate_image_description(const std::string& image_data) {
    // Decode base64 image data
    std::string decoded_image = base64_decode(image_data);

    // Convert decoded image to OpenCV Mat
    std::vector<uchar> image_vector(decoded_image.begin(), decoded_image.end());
    cv::Mat image = cv::imdecode(image_vector, cv::IMREAD_COLOR);

    if (image.empty()) {
        return "Error: Failed to decode image";
    }

    // Convert OpenCV Mat to clip_image_u8
    clip_image_u8 clip_image;
    clip_image.nx = image.cols;
    clip_image.ny = image.rows;
    clip_image.buf.resize(image.total() * 3);
    std::memcpy(clip_image.buf.data(), image.data, clip_image.buf.size());

    // Generate image embedding
    float* image_embed = nullptr;
    int n_img_pos = 0;
    if (!llava_image_embed_make_with_clip_img(clip_ctx, std::thread::hardware_concurrency(), &clip_image, &image_embed, &n_img_pos)) {
        return "Error: Failed to generate image embedding";
    }

    // Prepare prompt
    std::string prompt = "Describe the image in detail.";
    std::vector<llama_token> tokens = llama_tokenize(llama_ctx, prompt, true);

    // Generate description
    std::string description;
    int n_past = 0;
    llava_eval_image_embed(llama_ctx, &llava_image_embed{image_embed, n_img_pos}, 1, &n_past);

    for (int i = 0; i < 100; ++i) { // Generate up to 100 tokens
        llama_token id = llama_sample_token(llama_ctx);
        if (id == llama_token_eos()) break;

        const char* token_str = llama_token_to_str(llama_ctx, id);
        description += token_str;

        tokens.push_back(id);
        if (llama_decode(llama_ctx, llama_batch_get_one(&tokens.back(), 1, n_past, 0))) {
            break;
        }
        n_past += 1;
    }

    free(image_embed);
    return description;
}

std::string base64_decode(const std::string& encoded_string) {
    std::string decoded_string;
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];

    static const std::string base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    while (in_len-- && (encoded_string[in_] != '=') && (isalnum(encoded_string[in_]) || (encoded_string[in_] == '+') || (encoded_string[in_] == '/'))) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                decoded_string += char_array_3[i];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) decoded_string += char_array_3[j];
    }

    return decoded_string;
}

std::string process_request(const std::string& request_body) {
    rapidjson::Document d;
    d.Parse(request_body.c_str());

    if (d.HasParseError() || !d.IsObject()) {
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Invalid JSON\"}";
    }

    if (!d.HasMember("messages") || !d["messages"].IsArray()) {
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Missing or invalid 'messages' field\"}";
    }

    std::string system_message;
    std::string user_message;
    std::string image_data;

    for (const auto& message : d["messages"].GetArray()) {
        if (!message.IsObject() || !message.HasMember("role") || !message.HasMember("content")) {
            continue;
        }

        std::string role = message["role"].GetString();
        
        if (role == "system" && message["content"].IsString()) {
            system_message = message["content"].GetString();
        } else if (role == "user") {
            if (message["content"].IsArray()) {
                for (const auto& content : message["content"].GetArray()) {
                    if (content.HasMember("type") && content["type"].GetString() == std::string("text")) {
                        user_message = content["text"].GetString();
                    } else if (content.HasMember("type") && content["type"].GetString() == std::string("image_url")) {
                        std::string image_url = content["image_url"]["url"].GetString();
                        if (image_url.substr(0, 22) == "data:image/png;base64,") {
                            image_data = image_url.substr(22);
                        }
                    }
                }
            }
        }
    }

    if (image_data.empty()) {
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"No valid image data found\"}";
    }

    std::string description = generate_image_description(image_data, system_message, user_message);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("choices");
    writer.StartArray();
    writer.StartObject();
    writer.Key("message");
    writer.StartObject();
    writer.Key("role");
    writer.String("assistant");
    writer.Key("content");
    writer.String(description.c_str());
    writer.EndObject();
    writer.EndObject();
    writer.EndArray();
    writer.EndObject();

    std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n";
    response += buffer.GetString();

    return response;
}

std::string generate_image_description(const std::string& image_data, const std::string& system_message, const std::string& user_message) {
    // Decode base64 image data
    std::string decoded_image = base64_decode(image_data);

    // Convert decoded image to OpenCV Mat
    std::vector<uchar> image_vector(decoded_image.begin(), decoded_image.end());
    cv::Mat image = cv::imdecode(image_vector, cv::IMREAD_COLOR);

    if (image.empty()) {
        return "Error: Failed to decode image";
    }

    // Convert OpenCV Mat to clip_image_u8
    clip_image_u8 clip_image;
    clip_image.nx = image.cols;
    clip_image.ny = image.rows;
    clip_image.buf.resize(image.total() * 3);
    std::memcpy(clip_image.buf.data(), image.data, clip_image.buf.size());

    // Generate image embedding
    float* image_embed = nullptr;
    int n_img_pos = 0;
    if (!llava_image_embed_make_with_clip_img(clip_ctx, std::thread::hardware_concurrency(), &clip_image, &image_embed, &n_img_pos)) {
        return "Error: Failed to generate image embedding";
    }

    // Prepare prompt
    std::string prompt = system_message + "\n\nUser: " + user_message + "\n\nAssistant: ";
    std::vector<llama_token> tokens = llama_tokenize(llama_ctx, prompt, true);

    // Generate description
    std::string description;
    int n_past = 0;
    llava_eval_image_embed(llama_ctx, &llava_image_embed{image_embed, n_img_pos}, 1, &n_past);

    for (const auto& token : tokens) {
        if (llama_decode(llama_ctx, llama_batch_get_one(&token, 1, n_past, 0))) {
            break;
        }
        n_past += 1;
    }

    for (int i = 0; i < 500; ++i) { // Generate up to 500 tokens
        llama_token id = llama_sample_token(llama_ctx);
        if (id == llama_token_eos()) break;

        const char* token_str = llama_token_to_str(llama_ctx, id);
        description += token_str;

        if (llama_decode(llama_ctx, llama_batch_get_one(&id, 1, n_past, 0))) {
            break;
        }
        n_past += 1;
    }

    free(image_embed);
    return description;
}

// ... (rest of the code remains the same)