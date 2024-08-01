cmake_minimum_required(VERSION 3.12)
project(llava-server CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to use CUDA
option(LLAVA_CUDA "Enable CUDA support" OFF)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

# CUDA support
if (LLAVA_CUDA)
    enable_language(CUDA)
    
    # Method 1: Using PkgConfig
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(CUBLAS IMPORTED_TARGET)
        if(CUBLAS_FOUND)
            set(CUDA_LIBRARIES PkgConfig::CUBLAS)
        endif()
    endif()
    
    # Method 2: Using FindCUDAToolkit
    if(NOT CUBLAS_FOUND)
        find_package(CUDAToolkit REQUIRED)
        set(CUDA_LIBRARIES CUDA::cudart CUDA::cublas)
        
        # Additional settings for SDK environments
        set(CUDAToolkit_TARGET_DIR "${CUDA_TOOLKIT_TARGET_DIR}" CACHE PATH "" FORCE)
        find_library(CUDA_CUDART NAMES cudart HINTS ${CUDA_TOOLKIT_TARGET_DIR} PATH_SUFFIXES lib64 lib)
    endif()
    
    add_definitions(-DGGML_USE_CUBLAS)
endif()

# Add nlohmann/json
include_directories(./include)

# Add ggml library
add_library(ggml STATIC
    ../../ggml/src/ggml.c
    ../../ggml/src/ggml-alloc.c
    ../../ggml/src/ggml-backend.c
    ../../ggml/src/ggml-quants.c
)
target_include_directories(ggml PUBLIC 
    ../../ggml/include
    ../../ggml/src
)
if (LLAVA_CUDA)
    target_link_libraries(ggml PRIVATE ${CUDA_LIBRARIES})
endif()

# Add the llama library
add_library(llama OBJECT
    ../../src/llama.cpp
    ../../include/llama.h
)

target_include_directories(llama PUBLIC
    .
    ../..
    ../../common
    ../../include
    ../../ggml/include
    ../../ggml/src
)

target_link_libraries(llama PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama PRIVATE cxx_std_11)
if (LLAVA_CUDA)
    target_link_libraries(llama PRIVATE ${CUDA_LIBRARIES})
endif()

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

target_link_libraries(llava PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llava PRIVATE cxx_std_11)
if (LLAVA_CUDA)
    target_link_libraries(llava PRIVATE ${CUDA_LIBRARIES})
endif()

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
    ${CMAKE_THREAD_LIBS_INIT}
)

if (LLAVA_CUDA)
    target_link_libraries(llava-server PRIVATE ${CUDA_LIBRARIES})
endif()

target_include_directories(llava-server
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    .
    ../..
    ../../common
    ../../include
    ../../ggml/include
    ../../ggml/src
)

# Installation rules
install(TARGETS llava-server
    RUNTIME DESTINATION bin
)