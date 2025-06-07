cmake_minimum_required(VERSION 3.15)
project(musicgen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ggml)

set(PROJECT_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml
)

set(TESTS
    test_conv
    test_read_gguf
    test_seanet
    test_lstm
    test_quantizer
    test_mul_mat
    test_encoder
)

# Create each test executable and set includes + linking
foreach(TEST_NAME IN LISTS TESTS)
    add_executable(${TEST_NAME} tests/${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE ${PROJECT_INCLUDES})
    target_link_libraries(${TEST_NAME} PRIVATE ggml)
endforeach()
