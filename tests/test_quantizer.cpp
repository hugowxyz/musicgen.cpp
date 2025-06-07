#include "quantizer.h"
#include "utils.h"

#include <cstdlib>
#include <ctime>
#include <vector>
#include <cstdio>

void test_quantizer_encode() {
    // --- GGML setup ---
    const size_t ctx_size = 200 * 100000;
    void *ctx_data = malloc(ctx_size);
    struct ggml_context *ctx = ggml_init({ .mem_size = ctx_size, .mem_buffer = ctx_data });

    std::srand(std::time(nullptr));

    // --- dimensions ---
    const int seq_length   = 10;    // e.g. 10 time steps
    const int hidden_dim   = 16;   // D
    const int num_stages   = 2;     // n_q
    const int codebook_size = 8; // K

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

void test_ee() {
    // --- GGML setup ---
    const size_t ctx_size = 1024 * 1024;
    void *ctx_data = malloc(ctx_size);
    struct ggml_context *ctx = ggml_init({ .mem_size = ctx_size, .mem_buffer = ctx_data });

    // --- dimensions ---
    const int seq_length    = 2;
    const int hidden_dim    = 2;
    const int num_stages    = 1;
    const int codebook_size = 2;

    // --- known input ---
    std::vector<float> input_data = {
        1.0f, 0.0f,  // closer to codebook[0]
        0.0f, 1.0f   // closer to codebook[1]
    };
    struct ggml_tensor *encoded_inp = create_2d_tensor(
        ctx, input_data.data(), seq_length, hidden_dim
    );

    std::puts("Original input embeddings:");
    print_ggml_2d_tensor(encoded_inp);

    // --- build quantizer with known codebook ---
    quantizer quant;
    quant.blocks.resize(num_stages);

    // Codebook: index 0 = [1, 0], index 1 = [0, 1]
    std::vector<float> embed_data = {
        1.0f, 0.0f,  // code 0
        0.0f, 1.0f   // code 1
    };
    quant.blocks[0].embed = create_2d_tensor(
        ctx, embed_data.data(), codebook_size, hidden_dim
    );

    // --- encode ---
    struct ggml_tensor *codes = quantizer_encode(&quant, ctx, encoded_inp);
    struct ggml_tensor *computed_codes = compute_graph_from_tensor(ctx, codes, 1);

    std::puts("\nEncoded codes (should be 0 and 1):");
    print_ggml_2d_tensor(computed_codes);

    // --- decode ---
    struct ggml_tensor *decoded = quantizer_decode(&quant, ctx, computed_codes);
    struct ggml_tensor *reconstructed = compute_graph_from_tensor(ctx, decoded, 1);

    std::puts("\nReconstructed embeddings (should match codebook vectors):");
    print_ggml_2d_tensor(reconstructed);

    // --- cleanup ---
    ggml_free(ctx);
    free(ctx_data);
}

int main() {
    // test_quantizer_encode();
    test_ee();
    return 0;
}
