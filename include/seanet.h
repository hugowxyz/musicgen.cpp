#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "conv.h"

struct ggml_tensor * seanet_resnet_block(
    struct ggml_context * ctx,
    struct ggml_tensor  * input,            // [batch_size, in_ch, seq_len]
    struct ggml_tensor  * weight_3x1,       // [bottleneck_ch, in_ch, 3]
    struct ggml_tensor  * bias_3x1,         // [bottleneck_ch] or NULL
    struct ggml_tensor  * weight_1x1,       // [in_ch, bottleneck_ch, 1]
    struct ggml_tensor  * bias_1x1          // [in_ch] or NULL
) {

    struct ggml_tensor * act1 = ggml_elu(ctx, input);
    struct ggml_tensor * conv1 = streamable_conv1d(ctx, act1, weight_3x1, bias_3x1, 1, 1, 1);  // stride=1, padding=1
    struct ggml_tensor * act2 = ggml_elu(ctx, conv1);
    struct ggml_tensor * conv2 = streamable_conv1d(ctx, act2, weight_1x1, bias_1x1, 1, 0, 1);  // stride=1, padding=0
    struct ggml_tensor * output = ggml_add(ctx, input, conv2);

    return output;
}