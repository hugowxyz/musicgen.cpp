#include "ggml-cpu.h"
#include "ggml.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_ggml_conv1d() {
    const int in_ch = 1;
    const int seq_len = 5;
    const int kernel = 3;
    const int out_ch = 1;
    const int stride = 1;
    const int pad = 0;
    const int dil = 1;

    // Input and weight data
    float input_data[in_ch * seq_len] = {1, 1, 1, 1, 1};
    float weight_data[kernel * in_ch * out_ch] = {10, 10, 10}; // [K, IC, OC]
    float bias_data[1] = {1};

    size_t ctx_size = 1024 * 1024;
    struct ggml_init_params params = {ctx_size, NULL, false};
    struct ggml_context *ctx = ggml_init(params);

    // Input tensor: PyTorch [1, 1, 5] -> ggml [5, 1, 1]
    // struct ggml_tensor * input_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, in_ch, 1);
    struct ggml_tensor *input_f32 =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5, 1, 1);
    memcpy(input_f32->data, input_data, ggml_nbytes(input_f32) * 3);

    print_ggml_3d_tensor(input_f32);

    struct ggml_tensor *weights_f16 =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 1, 1);
    ggml_fp16_t *dst = (ggml_fp16_t *)weights_f16->data;
    for (int i = 0; i < 3; ++i) {
        dst[i] = ggml_fp32_to_fp16(weight_data[i]);
    }

    print_ggml_3d_tensor(weights_f16);

    struct ggml_tensor *bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1);
    memcpy(bias->data, bias_data, ggml_nbytes(weights_f16));

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

int main() {
    printf("Running test_ggml_conv1d\n");
    test_ggml_conv1d();
}