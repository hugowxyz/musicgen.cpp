cmake_minimum_required(VERSION 3.15)
project(musicgen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)

add_subdirectory(ggml)
add_subdirectory(include)

add_executable(test_conv tests/test_conv.cpp)
add_executable((test_gguf tests/test_gguf.cpp))

target_include_directories(test_conv PRIVATE include ggml/include ggml)
target_include_directories(test_gguf PRIVATE include ggml/include ggml)

target_link_libraries(test_conv PRIVATE ggml)
target_link_libraries(test_gguf PRIVATE ggml)

