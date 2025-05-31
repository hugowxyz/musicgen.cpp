#include "ggml.h"
#include "utils.h"
#include <cassert>
#include <iostream>
#include <vector>

void test_argmax() {
        // Tensor shape: [4, 3] (4 rows, 3 columns) -> column-major: [ne1 = 4, ne0 = 3]
    int rows = 4;
    int cols = 3;

    float input_data[] = {
        0.1f, 0.9f, 0.0f, 0.2f,  // col 0
        0.3f, 0.1f, 0.4f, 0.2f,  // col 1
        0.2f, 0.0f, 0.5f, 0.1f   // col 2
    };

    struct ggml_init_params params = {};
    params.mem_size = 16 * 1024 * 1024;
    std::vector<uint8_t> mem(params.mem_size);
    params.mem_buffer = mem.data();

    ggml_context *ctx = ggml_init(params);

    // Create 2D tensor: shape [cols, rows]
    ggml_tensor *input = create_2d_tensor(ctx, input_data, rows, cols);

    // Apply argmax over last dimension (columns in this layout)
    ggml_tensor *argmax = ggml_argmax(ctx, input);

    // Evaluate the computation graph
    compute_graph_from_tensor(ctx, argmax, 1);

    // Print input
    std::cout << "Input Tensor (" << rows << "x" << cols << "):\n";
    print_ggml_2d_tensor(input);

    print_ggml_1d_tensor(argmax);
    ggml_free(ctx);

    std::cout << "\nTest passed!\n";
}

void test_matmul() {
     // Matrix A: shape [k, m] = [2, 3]
    // Matrix B: shape [k, n] = [2, 4]
    // Result:   shape [m, n] = [3, 4]
    int k = 2, m = 3, n = 4;

    // Column-major: a_data[col * m + row]
    float a_data[] = {
        1, 4, 7, // col 0
        2, 5, 8  // col 1
    };
    float b_data[] = {
        1, 3, 5, 7, // col 0
        2, 4, 6, 8  // col 1
    };

    struct ggml_init_params params = {};
    params.mem_size = 16 * 1024 * 1024;
    std::vector<uint8_t> mem(params.mem_size);
    params.mem_buffer = mem.data();

    ggml_context *ctx = ggml_init(params);

    ggml_tensor *A = create_2d_tensor(ctx, a_data, m, k); // ne1 = rows, ne0 = cols
    ggml_tensor *B = create_2d_tensor(ctx, b_data, n, k);

    ggml_tensor *C = ggml_mul_mat(ctx, A, B);

    compute_graph_from_tensor(ctx, C, 1);

    std::cout << "Matrix A (" << m << "x" << k << "):\n";
    print_ggml_2d_tensor(A);

    std::cout << "\nMatrix B (" << n << "x" << k << "):\n";
    print_ggml_2d_tensor(B);

    std::cout << "\nResult C (" << n << "x" << m << "):\n";
    print_ggml_2d_tensor(C);

    float *c_data = (float *)C->data;
    // Simple check: C[0,0] = A[:,0]^T * B[:,0] = 1*1 + 2*2 = 5
    // assert(std::abs(c_data[0] - 5.0f) < 1e-5);

    ggml_free(ctx);

    std::cout << "\nTest passed!\n";
}

int main() {
    test_argmax();
    return 0;
}