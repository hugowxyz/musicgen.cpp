#include "seanet.h"
#include "utils.h"

void test_seanet_resnet_block_identity_pass_through() {
    const size_t ctx_size = 16 * 1024;
    void * ctx_data = malloc(ctx_size);
    struct ggml_context * ctx = ggml_init(ggml_init_params { .mem_size = ctx_size, .mem_buffer = ctx_data });

    // Example: batch=1, in_ch=2, seq_len=4, bottleneck_ch=2
    int batch_size = 1, in_ch = 2, seq_len = 4, bottleneck_ch = 2;

    // Create input tensor [1, 2, 4] filled with 1.0f
    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq_len, in_ch, batch_size);
    for (int i = 0; i < seq_len * in_ch * batch_size; i++) {
        ggml_set_f32_1d(input, i, 1.0f);
    }

    // Create zeroed weights and biases
    struct ggml_tensor * weight_3x1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, in_ch, bottleneck_ch);
    struct ggml_tensor * bias_3x1   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bottleneck_ch);
    struct ggml_tensor * weight_1x1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, bottleneck_ch, in_ch);
    struct ggml_tensor * bias_1x1   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_ch);

    // All weights and biases are zero-initialized by default, but set explicitly to be sure
    for (int i = 0; i < 3 * in_ch * bottleneck_ch; i++) ggml_set_f32_1d(weight_3x1, i, 0.0f);
    for (int i = 0; i < bottleneck_ch; i++) ggml_set_f32_1d(bias_3x1, i, 0.0f);
    for (int i = 0; i < bottleneck_ch * in_ch; i++) ggml_set_f32_1d(weight_1x1, i, 0.0f);
    for (int i = 0; i < in_ch; i++) ggml_set_f32_1d(bias_1x1, i, 0.0f);

    // Run ResNet block
    struct ggml_tensor * output = seanet_resnet_block(ctx, input, weight_3x1, bias_3x1, weight_1x1, bias_1x1);

    // Validate output shape matches input
    assert(output->ne[0] == seq_len);
    assert(output->ne[1] == in_ch);
    assert(output->ne[2] == batch_size);

    // Validate output values equal input values (since conv is zeroed)
    for (int i = 0; i < seq_len * in_ch * batch_size; i++) {
        float value = ggml_get_f32_1d(output, i);
        printf("output[%d] = %f\n", i, value);
        assert(value == 1.0f);
    }

    printf("Test passed.\n");

    ggml_free(ctx);
    free(ctx_data);
}



int main() {
    test_seanet_resnet_block_identity_pass_through();
}