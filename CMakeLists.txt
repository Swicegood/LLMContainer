cmake_minimum_required(VERSION 3.12)
project(llava-server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to use CUDA
option(LLAVA_CUDA "Enable CUDA support" OFF)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(Threads REQUIRED)

# Add the llama library
add_library(llama OBJECT
    ../../src/llama.cpp
    ../../include/llama.h
)

target_include_directories(llama PUBLIC
    .
    ../..
    ../../common
)

target_link_libraries(llama PRIVATE ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama PRIVATE cxx_std_11)

# Add the llava library
add_library(llava OBJECT
    ./llava.cpp
    ./llava.h
    ./clip.cpp
    ./clip.h
)

target_include_directories(llava PUBLIC
    .
    ../..
    ../../common
)

target_link_libraries(llava PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llava PRIVATE cxx_std_11)

# Add the server executable
add_executable(llava-server
    llava-server.cpp
)

target_link_libraries(llava-server
    PRIVATE
    llama
    llava
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
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

# CUDA support
if (LLAVA_CUDA)
    add_definitions(-DGGML_USE_CUBLAS)
    find_package(CUDA REQUIRED)
    target_link_libraries(llama PRIVATE CUDA::cudart CUDA::cublas)
    target_link_libraries(llava PRIVATE CUDA::cudart CUDA::cublas)
    target_link_libraries(llava-server PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Installation rules
install(TARGETS llava-server
    RUNTIME DESTINATION bin
)