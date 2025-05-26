#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "conv.h"

// SEANet residual block with 2×ELU, weight-normalized Conv1D, identity shortcut
struct ggml_tensor * seanet_resnet_block(
    struct ggml_context * ctx,
    struct ggml_tensor  * input,        // [B, in_ch, T]
    // Parameters for first conv (kernel size 3)
    struct ggml_tensor  * weight_g1,    // [bottleneck_ch]
    struct ggml_tensor  * weight_v1,    // [3, in_ch, bottleneck_ch]
    struct ggml_tensor  * bias1,        // [bottleneck_ch] or NULL
    // Parameters for second conv (kernel size 1)
    struct ggml_tensor  * weight_g2,    // [out_ch]
    struct ggml_tensor  * weight_v2,    // [1, bottleneck_ch, out_ch]
    struct ggml_tensor  * bias2         // [out_ch] or NULL
) {
    // 1st activation + bottleneck conv
    struct ggml_tensor * act1  = ggml_elu(ctx, input);
    struct ggml_tensor * conv1 = streamable_conv1d_wn(
        ctx,
        act1,
        weight_g1,
        weight_v1,
        bias1,
        /*stride=*/1,
        /*padding=*/1,
        /*dilation=*/1
    );

    // 2nd activation + expansion conv
    struct ggml_tensor * act2  = ggml_elu(ctx, conv1);
    struct ggml_tensor * conv2 = streamable_conv1d_wn(
        ctx,
        act2,
        weight_g2,
        weight_v2,
        bias2,
        /*stride=*/1,
        /*padding=*/0,
        /*dilation=*/1
    );

    // Identity shortcut: input + output of conv2
    struct ggml_tensor * output = ggml_add(ctx, input, conv2);
    return output;
}
