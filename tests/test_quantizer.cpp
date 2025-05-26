#include "quantizer.h"
#include "utils.h"

#include <cstdlib>
#include <ctime>
#include <vector>
#include <cstdio>

void test_quantizer_encode() {
    // --- GGML setup ---
    const size_t ctx_size = 50 * 100000;
    void *ctx_data = malloc(ctx_size);
    struct ggml_context *ctx = ggml_init({ .mem_size = ctx_size, .mem_buffer = ctx_data });

    std::srand(std::time(nullptr));

    // --- dimensions ---
    const int seq_length   = 10;    // e.g. 10 time steps
    const int hidden_dim   = 128;   // D
    const int num_stages   = 4;     // n_q
    const int codebook_size = 2048; // K

    // --- random input ---
    std::vector<float> input_data(seq_length * hidden_dim);
    for (auto &x : input_data) {
        x = (float(std::rand()) / RAND_MAX) * 2 - 1;
    }
    // [seq_length, D]
    struct ggml_tensor *encoded_inp = create_2d_tensor(
        ctx, input_data.data(), seq_length, hidden_dim
    );

    // --- build quantizer ---
    quantizer quant;
    quant.blocks.resize(num_stages);
    for (int i = 0; i < num_stages; ++i) {
        // random codebook embed [K, D]
        std::vector<float> embed_data(codebook_size * hidden_dim);
        for (auto &w : embed_data) {
            w = (float(std::rand()) / RAND_MAX) * 2 - 1;
        }
        quant.blocks[i].embed = create_2d_tensor(
            ctx, embed_data.data(), codebook_size, hidden_dim
        );
    }

    // --- encode ---
    struct ggml_tensor *codes = quantizer_encode(
        &quant, ctx, encoded_inp
    );

    // --- run graph ---
    struct ggml_tensor *out = compute_graph_from_tensor(
        ctx, codes, /*n_threads=*/1
    );

    // --- print codes [seq_length × num_stages] ---
    print_ggml_2d_tensor(out);

    // --- cleanup ---
    ggml_free(ctx);
    free(ctx_data);
}

int main() {
    test_quantizer_encode();
    return 0;
}
