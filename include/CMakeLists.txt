add_library(encodec INTERFACE)
target_include_directories(encodec INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include
) 