cmake_minimum_required(VERSION 3.10)
project(llava-server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(RapidJSON REQUIRED)

add_executable(llava-server
    llava-server.cpp
    clip.cpp
    llama.cpp
    llava.cpp
)

target_include_directories(llava-server PRIVATE ${OpenCV_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS} ${RAPIDJSON_INCLUDE_DIRS})
target_link_libraries(llava-server PRIVATE ${OpenCV_LIBS} ${CURL_LIBRARIES} pthread)