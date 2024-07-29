cmake_minimum_required(VERSION 3.12)
project(llava-server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to use CUDA (from original)
option(LLAVA_CUDA "Enable CUDA support" OFF)

target_include_directories(llama PUBLIC .)
target_include_directories(llama PUBLIC ../..)
target_include_directories(llama PUBLIC ../../common)

# Add the llama library (from original)
add_library(llama OBJECT
            llama.cpp
            llama.h
            )

target_link_libraries(llama PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(llama PRIVATE cxx_std_11)

# Add the llava library (from original)
add_library(llava OBJECT
            llava.cpp
            llava.h
            clip.cpp
            clip.h
            )

target_link_libraries(llava PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(llava PUBLIC .)
target_include_directories(llava PUBLIC ../..)
target_include_directories(llava PUBLIC ../../common)

target_compile_features(llava PRIVATE cxx_std_11)

# New additions for the server
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(RapidJSON REQUIRED)

# Add the server executable
add_executable(llava-server
    llava-server.cpp
)

target_link_libraries(llava-server 
    PRIVATE 
    llama 
    llava 
    ggml 
    ${OpenCV_LIBS} 
    ${CURL_LIBRARIES} 
    pthread
)

target_include_directories(llava-server 
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS} 
    ${CURL_INCLUDE_DIRS} 
    ${RAPIDJSON_INCLUDE_DIRS}
    .
    ../..
    ../../common
)

# CUDA support (from original)
if (LLAVA_CUDA)
    add_definitions(-DGGML_USE_CUBLAS)
    find_package(CUDA REQUIRED)
    target_link_libraries(llama PRIVATE CUDA::cudart CUDA::cublas)
    target_link_libraries(llava PRIVATE CUDA::cudart CUDA::cublas)
    target_link_libraries(llava-server PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Installation rules (if needed)
install(TARGETS llava-server
    RUNTIME DESTINATION bin
)