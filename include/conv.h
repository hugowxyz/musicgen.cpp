#pragma once

#include "ggml.h"

struct ggml_tensor * streamable_conv1d(
    struct ggml_context * ctx,
    struct ggml_tensor  * input,    // [batch_size, in_channels, seq_len]
    struct ggml_tensor  * weights,  // [kernel_size, in_channels, out_channels]
    struct ggml_tensor  * bias,     // [out_channels] or NULL
    int                   stride,
    int                   padding,
    int                   dilation) {

    // If weights are not FP16, convert to FP16
    if (weights->type != GGML_TYPE_F16) {
        int64_t ne0 = weights->ne[0]; // kernel_size
        int64_t ne1 = weights->ne[1]; // in_channels
        int64_t ne2 = weights->ne[2]; // out_channels

        struct ggml_tensor * weights_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, ne0, ne1, ne2);
        float * src = (float *)weights->data;
        ggml_fp16_t * dst = (ggml_fp16_t *)weights_f16->data;

        for (int64_t i = 0; i < ne0 * ne1 * ne2; ++i) {
            dst[i] = ggml_fp32_to_fp16(src[i]);
        }

        weights = weights_f16;
    }

    struct ggml_tensor * conv_output = ggml_conv_1d(ctx, weights, input, stride, padding, dilation);

    if (bias != NULL) {
        conv_output = ggml_add(ctx, conv_output, ggml_repeat(ctx, bias, conv_output));
    }

    return conv_output;
}