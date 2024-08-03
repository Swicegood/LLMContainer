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

#include <nlohmann/json.hpp>

#undef CLIP_H

#include "clip.h"
#include "llama.h"
#include "llava.h"

using json = nlohmann::json;

// Global variables
struct clip_ctx *clip_ctx;
llama_model *llama_model;
llama_context *llama_ctx;

// Forward declarations
std::string process_request(const std::string &request_body);
std::string generate_image_description(const std::string &image_data, const std::string &system_message, const std::string &user_message);
std::string base64_decode(const std::string &encoded_string);

// Main function
int main(int argc, char *argv[])
{
    // Parse command line arguments
    std::string model_path, mmproj_path;
    int port = 8080;

    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--model" && i + 1 < argc)
        {
            model_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--mmproj" && i + 1 < argc)
        {
            mmproj_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--port" && i + 1 < argc)
        {
            port = std::stoi(argv[++i]);
        }
    }

    if (model_path.empty() || mmproj_path.empty())
    {
        std::cerr << "Usage: " << argv[0] << " --model <path> --mmproj <path> [--port <port>]" << std::endl;
        return 1;
    }

    // Initialize CLIP
    clip_ctx = clip_model_load(mmproj_path.c_str(), 1);
    if (!clip_ctx)
    {
        std::cerr << "Failed to load CLIP model" << std::endl;
        return 1;
    }

    // Initialize LLaMA
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!llama_model)
    {
        std::cerr << "Failed to load LLaMA model" << std::endl;
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    llama_ctx = llama_new_context_with_model(llama_model, ctx_params);
    if (!llama_ctx)
    {
        std::cerr << "Failed to create LLaMA context" << std::endl;
        return 1;
    }

    // Set up server socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1)
    {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        std::cerr << "Failed to bind to port " << port << std::endl;
        return 1;
    }

    if (listen(server_fd, 3) < 0)
    {
        std::cerr << "Failed to listen on socket" << std::endl;
        return 1;
    }

    std::cout << "Server listening on port " << port << std::endl;

    while (true)
    {
        int client_socket = accept(server_fd, nullptr, nullptr);
        if (client_socket < 0)
        {
            std::cerr << "Failed to accept client connection" << std::endl;
            continue;
        }

        // Handle client in a separate thread
        std::thread([client_socket]()
                    {
            char buffer[1024] = {0};
            std::string request;
        std::string request_headers;
        std::string request_body;
        int content_length = 0;

        while (true) {
            int valread = read(client_socket, buffer, 1024);
            if (valread <= 0) break;
            request_headers.append(buffer, valread);
            
            // Check if we've reached the end of the headers
            size_t header_end = request_headers.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                // Extract the Content-Length
                size_t content_length_pos = request_headers.find("Content-Length: ");
                if (content_length_pos != std::string::npos) {
                    content_length_pos += 16; // move past "Content-Length: "
                    size_t content_length_end = request_headers.find("\r\n", content_length_pos);
                    content_length = std::stoi(request_headers.substr(content_length_pos, content_length_end - content_length_pos));
                }
                
                // Move any body content to the request_body string
                request_body = request_headers.substr(header_end + 4);
                request_headers = request_headers.substr(0, header_end);
                break;
            }
        }

        // Read the rest of the body if necessary
        while (request_body.length() < content_length) {
            int remaining = content_length - request_body.length();
            int valread = read(client_socket, buffer, std::min(remaining, 1024));
            if (valread <= 0) break;
            request_body.append(buffer, valread);
        }


            std::string response = process_request(request_body);

            send(client_socket, response.c_str(), response.length(), 0);
            close(client_socket); })
            .detach();

        return 0;
    }
}

std::string process_request(const std::string &request_body)
{
    json request;
    try
    {
        request = json::parse(request_body);
    }
    catch (json::parse_error &e)
    {
        std::string error_msg = "Invalid JSON: " + std::string(e.what()) + ". Request body: " + request_body;
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"" + error_msg + "\"}";
    }

    if (!request.contains("messages") || !request["messages"].is_array())
    {
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"Missing or invalid 'messages' field\"}";
    }

    std::string system_message;
    std::string user_message;
    std::string image_data;

    for (const auto &message : request["messages"])
    {
        if (!message.is_object() || !message.contains("role") || !message.contains("content"))
        {
            continue;
        }

        std::string role = message["role"];

        if (role == "system" && message["content"].is_string())
        {
            system_message = message["content"];
        }
        else if (role == "user")
        {
            if (message["content"].is_array())
            {
                for (const auto &content : message["content"])
                {
                    if (content.contains("type") && content["type"] == "text")
                    {
                        user_message = content["text"];
                    }
                    else if (content.contains("type") && content["type"] == "image_url")
                    {
                        std::string image_url = content["image_url"]["url"];
                        if (image_url.substr(0, 22) == "data:image/png;base64,")
                        {
                            image_data = image_url.substr(22);
                        }
                    }
                }
            }
        }
    }

    if (image_data.empty())
    {
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\": \"No valid image data found\"}";
    }

    std::string description = generate_image_description(image_data, system_message, user_message);

    json response = {
        {"choices", {{{"message", {{"role", "assistant"}, {"content", description}}}}}}};

    std::string response_body = response.dump();
    std::string http_response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" + response_body;

    return http_response;
}

std::string base64_decode(const std::string &encoded_string)
{
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

    while (in_len-- && (encoded_string[in_] != '=') && (isalnum(encoded_string[in_]) || (encoded_string[in_] == '+') || (encoded_string[in_] == '/')))
    {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4)
        {
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

    if (i)
    {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++)
            decoded_string += char_array_3[j];
    }

    return decoded_string;
}

std::string generate_image_description(const std::string &image_data, const std::string &system_message, const std::string &user_message)
{
    // Decode base64 image data
    std::string decoded_image = base64_decode(image_data);

    // Convert decoded image to OpenCV Mat
    std::vector<uchar> image_vector(decoded_image.begin(), decoded_image.end());
    cv::Mat image = cv::imdecode(image_vector, cv::IMREAD_COLOR);

    if (image.empty())
    {
        return "Error: Failed to decode image";
    }

    // Create clip_image_u8
    struct clip_image_u8 *clip_image = clip_image_u8_init();
    if (!clip_image)
    {
        return "Error: Failed to initialize clip_image_u8";
    }

    // Convert OpenCV Mat to clip_image_u8
    if (!clip_image_load_from_bytes(image.data, image.total() * image.elemSize(), clip_image))
    {
        clip_image_u8_free(clip_image);
        return "Error: Failed to load image data into clip_image_u8";
    }

    // Generate image embedding
    float *image_embed = nullptr;
    int n_img_pos = 0;
    if (!llava_image_embed_make_with_clip_img(clip_ctx, std::thread::hardware_concurrency(), clip_image, &image_embed, &n_img_pos))
    {
        clip_image_u8_free(clip_image);
        return "Error: Failed to generate image embedding";
    }

    // Prepare prompt
    std::string prompt = system_message + "\n\nUser: " + user_message + "\n\nAssistant: ";

    // Tokenize the prompt
    std::vector<llama_token> tokens(1024); // Pre-allocate space for tokens
    int n_tokens = llama_tokenize(llama_model,
                                  prompt.c_str(),
                                  prompt.length(),
                                  tokens.data(),
                                  tokens.size(),
                                  true,  // add_special
                                  true); // parse_special
    if (n_tokens < 0)
    {
        clip_image_u8_free(clip_image);
        free(image_embed);
        return "Error: Failed to tokenize prompt";
    }
    tokens.resize(n_tokens);

    // Generate description
    std::string description;
    int n_past = 0;

    // Create a named llava_image_embed object
    llava_image_embed img_embed = {image_embed, n_img_pos};

    // Use the address of the named object
    llava_eval_image_embed(llama_ctx, &img_embed, 1, &n_past);

    // Process tokens using indexing instead of range-based for loop
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        llama_batch batch = llama_batch_get_one(&tokens[i], 1, n_past, 0);
        if (llama_decode(llama_ctx, batch))
        {
            clip_image_u8_free(clip_image);
            free(image_embed);
            return "Error: Failed to decode tokens";
        }
        n_past++;
    }

    llama_token id = 0;
    char token_buf[8]; // Buffer to store the token piece, adjust size if needed
    for (int i = 0; i < 500; ++i)
    { // Generate up to 500 tokens
        llama_token_data_array candidates = {NULL, 0, false};
        id = llama_sample_token(llama_ctx, &candidates);

        if (llama_token_eos(llama_model) == id)
        {
            break;
        }

        int token_length = llama_token_to_piece(llama_model, id, token_buf, sizeof(token_buf), 0, false);
        if (token_length > 0)
        {
            description.append(token_buf, token_length);
        }

        llama_batch batch = llama_batch_get_one(&id, 1, n_past, 0);
        if (llama_decode(llama_ctx, batch))
        {
            break;
        }
        n_past++;
    }

    clip_image_u8_free(clip_image);
    free(image_embed);
    return description;
}