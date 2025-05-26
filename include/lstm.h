#pragma once
#include "ggml.h"

struct lstm_state {
    struct ggml_tensor * h_t;  // Hidden state
    struct ggml_tensor * c_t;  // Cell state
};

struct lstm_state lstm_step(
    struct ggml_context * ctx,
    struct ggml_tensor  * x_t,             // [D]
    struct ggml_tensor  * h_prev,          // [H]
    struct ggml_tensor  * c_prev,          // [H]
    struct ggml_tensor  * weight_ih_l0,    // [4H, D]
    struct ggml_tensor  * weight_hh_l0,    // [4H, H]
    struct ggml_tensor  * bias_ih_l0,      // [4H]
    struct ggml_tensor  * bias_hh_l0       // [4H]
) {
    // fused matmuls + add both biases
    struct ggml_tensor * WX    = ggml_mul_mat(ctx, weight_ih_l0, x_t);
    struct ggml_tensor * UH    = ggml_mul_mat(ctx, weight_hh_l0, h_prev);
    struct ggml_tensor * B     = ggml_add   (ctx, bias_ih_l0, bias_hh_l0);
    struct ggml_tensor * gates = ggml_add   (ctx, ggml_add(ctx, WX, UH), B);
    
    // compute H = (4H)/4
    const int total = bias_ih_l0->ne[0];   // = 4*H
    const int H     = total / 4;
    
    // carve out each gate pre-activation with view_1d
    const size_t stride = gates->nb[0];    // byte-stride per element
    struct ggml_tensor * i_p = ggml_view_1d(ctx, gates, H, (size_t)0 * H * stride);
    struct ggml_tensor * f_p = ggml_view_1d(ctx, gates, H, (size_t)1 * H * stride);
    struct ggml_tensor * g_p = ggml_view_1d(ctx, gates, H, (size_t)2 * H * stride);
    struct ggml_tensor * o_p = ggml_view_1d(ctx, gates, H, (size_t)3 * H * stride);
    
    // activations
    struct ggml_tensor * i_t = ggml_sigmoid(ctx, i_p);
    struct ggml_tensor * f_t = ggml_sigmoid(ctx, f_p);
    struct ggml_tensor * g_t = ggml_tanh   (ctx, g_p);
    struct ggml_tensor * o_t = ggml_sigmoid(ctx, o_p);
    
    // cell & hidden updates
    struct ggml_tensor * c_t = ggml_add(ctx,
        ggml_mul(ctx, f_t, c_prev),
        ggml_mul(ctx, i_t, g_t)
    );
    struct ggml_tensor * h_t = ggml_mul(ctx, o_t, ggml_tanh(ctx, c_t));
    
    return { h_t, c_t };
}
