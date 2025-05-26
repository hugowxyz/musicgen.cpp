#pragma once

#include "ggml.h"
#include <math.h>
#include <cstring>

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
        conv_output = ggml_transpose(ctx, conv_output); // [B, T, C]
        conv_output = ggml_add(ctx, ggml_repeat(ctx, bias, conv_output), conv_output);
        conv_output = ggml_cont(ctx, ggml_transpose(ctx, conv_output)); // back to [B, C, T], contiguous
    }

    return conv_output;
}


// weight normed conv1d
struct ggml_tensor * streamable_conv1d_wn(
    struct ggml_context * ctx,
    struct ggml_tensor  * input,      // [batch, in_ch, seq_len]
    struct ggml_tensor  * weight_g,   // [out_ch]
    struct ggml_tensor  * weight_v,   // [ks, in_ch, out_ch]
    struct ggml_tensor  * bias,       // [out_ch] or NULL
    int                   stride,
    int                   padding,
    int                   dilation) {

    // dims
    int64_t ks  = weight_v->ne[0];
    int64_t ic  = weight_v->ne[1];
    int64_t oc  = weight_v->ne[2];

    // Allocate a float buffer for norm and w
    float * gv = (float*) weight_g->data;      // length oc
    float * vv = (float*) weight_v->data;      // length ks*ic*oc

    float * norm = (float *)malloc(sizeof(float)*oc);
    float * wdata = (float *)malloc(sizeof(float)*ks*ic*oc);

    // Compute per‐channel norm
    for (int64_t j = 0; j < oc; ++j) {
        double sum2 = 0.0;
        int64_t base = j;
        for (int64_t i = 0; i < ks*ic; ++i) {
            float v_ijk = vv[ i*oc + base ];
            sum2 += (double)v_ijk * v_ijk;
        }
        norm[j] = sqrt(sum2) + 1e-6f;  // eps to avoid div0
    }

    // Build normalized weight w = g * v / norm
    for (int64_t idx = 0; idx < ks*ic*oc; ++idx) {
        int64_t j = idx % oc;
        wdata[idx] = gv[j] * ( vv[idx] / norm[j] );
    }

    struct ggml_tensor * w_fp32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ks, ic, oc);
    memcpy(w_fp32->data, wdata, sizeof(float)*ks*ic*oc);

    free(norm);
    free(wdata);

    struct ggml_tensor * out = streamable_conv1d(
        ctx,
        input,
        w_fp32,
        bias,
        stride,
        padding,
        dilation
    );

    return out;
}
