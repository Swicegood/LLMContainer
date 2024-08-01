cmake_minimum_required(VERSION 3.14)
project(llava-server CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(LLAVA_CUDA "Enable CUDA support" OFF)
option(GGML_STATIC "ggml: static link libraries" OFF)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

if (LLAVA_CUDA)
    enable_language(CUDA)
   
    # Method 1: Using PkgConfig
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(CUBLAS cublas IMPORTED_TARGET)
        if(CUBLAS_FOUND)
            set(CUDA_LIBRARIES PkgConfig::cublas)
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
set(GGML_SOURCES
    ../../ggml/src/ggml.c
    ../../ggml/src/ggml-alloc.c
    ../../ggml/src/ggml-backend.c
    ../../ggml/src/ggml-quants.c
)

if(LLAVA_CUDA)
    set(GGML_SOURCES ${GGML_SOURCES} ../../ggml/src/ggml-cuda.cu)
endif()

add_library(ggml STATIC ${GGML_SOURCES})

set_target_properties(ggml PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

target_compile_options(ggml PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wno-unused-function>
    $<$<COMPILE_LANGUAGE:C>:-Wno-unused-variable>
)

if (LLAVA_CUDA)
    set_source_files_properties(../../ggml/src/ggml-cuda.cu PROPERTIES LANGUAGE CUDA)
    target_compile_options(ggml PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

target_include_directories(ggml PUBLIC 
    ../../ggml/include
    ../../ggml/src
)

if (LLAVA_CUDA)
    target_link_libraries(ggml PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Add the llama library
add_library(llama
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
    target_link_libraries(llama PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Add the llava library
add_library(llava
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
    target_link_libraries(llava PRIVATE CUDA::cudart CUDA::cublas)
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

# if (LLAVA_CUDA)
#     target_link_libraries(llava-server PRIVATE CUDA::cudart CUDA::cublas)
# endif()

# target_include_directories(llava-server
#     PRIVATE
#     ${OpenCV_INCLUDE_DIRS}
#     ${CURL_INCLUDE_DIRS}
#     .
#     ../..
#     ../../common
#     ../../include
#     ../../ggml/include
#     ../../ggml/src
# )

# # Installation rules
# install(TARGETS llava-server
#     RUNTIME DESTINATION bin
# )