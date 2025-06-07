#include "ggml-cpu.h"
#include "ggml.h"
#include "utils.h"
#include "conv.h" // Your header file for streamable_conv1d_wn and streamable_conv1d

#include <vector>
#include <cstdlib>
#include <ctime>

#include <stdio.h>
#include <string.h>

void test_ggml_conv1d() {
    const int in_ch = 1;
    const int seq_len = 5;
    const int kernel = 3;
    const int out_ch = 1;
    const int stride = 1;
    const int pad = 0;
    const int dil = 1;
    const int batch = 1;

    // Input and weight data
    float input_data[in_ch * seq_len] = {1, 1, 1, 1, 1};
    float weight_data[kernel * in_ch * out_ch] = {10, 10, 10}; // [K, IC, OC]
    float bias_data[out_ch] = {1};

    // Initialize context
    size_t ctx_size = 1024 * 1024;
    struct ggml_init_params params = {ctx_size, NULL, false};
    struct ggml_context *ctx = ggml_init(params);

    // Tensor creation
    struct ggml_tensor *input_f32 =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, in_ch, batch);

    struct ggml_tensor *weights_f16 =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kernel, in_ch, out_ch);

    struct ggml_tensor *bias =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F16, out_ch);

    // Populate tensors
    memcpy(input_f32->data, input_data, ggml_nbytes(input_f32));

    ggml_fp16_t *dst = (ggml_fp16_t *)weights_f16->data;
    for (int i = 0; i < kernel * in_ch * out_ch; ++i) {
        dst[i] = ggml_fp32_to_fp16(weight_data[i]);
    }

    memcpy(bias->data, bias_data, ggml_nbytes(bias));

    // Debug prints
    print_ggml_3d_tensor(input_f32);
    print_ggml_3d_tensor(weights_f16);
    print_ggml_1d_tensor(bias);

    // Run 1D convolution
    struct ggml_tensor *output_fp16 =
        ggml_conv_1d(ctx, weights_f16, input_f32, stride, pad, dil);

    // Build and compute graph
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output_fp16);
    ggml_graph_compute_with_ctx(ctx, gf, -1);

    print_ggml_3d_tensor(output_fp16);

    ggml_free(ctx);
}

void test_streamable_conv1d_wn() {
    // GGML context setup
    const size_t ctx_size = 8 * 1024 * 1024;
    void *ctx_data = malloc(ctx_size);
    struct ggml_context *ctx = ggml_init({.mem_size = ctx_size, .mem_buffer = ctx_data});

    std::srand(std::time(nullptr));

    // Dimensions
    const int B = 1;   // batch size
    const int IC = 3;  // input channels
    const int OC = 2;  // output channels
    const int T = 5;   // sequence length
    const int KS = 3;  // kernel size

    const int STRIDE = 1;
    const int PADDING = 0;
    const int DILATION = 1;

    // Fill with random data
    auto fill_rand = [](std::vector<float> &v) {
        for (auto &x : v)
            x = (float(std::rand()) / RAND_MAX) * 2.f - 1.f;
    };

    std::vector<float> input_data(B * IC * T);
    std::vector<float> weight_v_data(KS * IC * OC);
    std::vector<float> weight_g_data(OC);
    std::vector<float> bias_data(OC);

    fill_rand(input_data);
    fill_rand(weight_v_data);
    fill_rand(weight_g_data);
    fill_rand(bias_data);

    // Build GGML tensors
    auto *input     = create_3d_tensor(ctx, input_data.data(), B, IC, T);             // [B, IC, T]
    auto *weight_v  = create_3d_tensor(ctx, weight_v_data.data(), OC, IC, KS);        // [KS, IC, OC]
    auto *weight_g  = create_1d_tensor(ctx, weight_g_data.data(), OC);                // [OC]
    auto *bias      = create_1d_tensor(ctx, bias_data.data(), OC);                    // [OC]

    // Run conv
    struct ggml_tensor *output = streamable_conv1d_wn(
        ctx, input, weight_g, weight_v, bias, STRIDE, PADDING, DILATION);

    // Compute and print result
    auto *computed = compute_graph_from_tensor(ctx, output, -1);
    print_ggml_3d_tensor(computed);

    // Cleanup
    ggml_free(ctx);
    free(ctx_data);
}


int main() {
    printf("Running test_ggml_conv1d\n");
    // test_ggml_conv1d();
    test_streamable_conv1d_wn();
}