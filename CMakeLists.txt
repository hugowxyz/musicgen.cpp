cmake_minimum_required(VERSION 3.15)
project(musicgen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ggml)

add_executable(test_conv tests/test_conv.cpp)
target_include_directories(test_conv PRIVATE include ggml/include ggml)
target_link_libraries(test_conv PRIVATE ggml)

add_executable(test_read_gguf tests/test_read_gguf.cpp)
target_include_directories(test_read_gguf PRIVATE include ggml/include ggml)
target_link_libraries(test_read_gguf PRIVATE ggml)

add_executable(test_seanet tests/test_seanet.cpp)
target_include_directories(test_seanet PRIVATE include ggml/include ggml)
target_link_libraries(test_seanet PRIVATE ggml)

add_executable(test_lstm tests/test_lstm.cpp)
target_include_directories(test_lstm PRIVATE include ggml/include ggml)
target_link_libraries(test_lstm PRIVATE ggml)

add_executable(test_quantizer tests/test_quantizer.cpp)
target_include_directories(test_quantizer PRIVATE include ggml/include ggml)
target_link_libraries(test_quantizer PRIVATE ggml)

