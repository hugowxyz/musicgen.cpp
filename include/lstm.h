#pragma once
#include "ggml.h"


struct lstm_state {
    struct ggml_tensor * h_t;  // Hidden state
    struct ggml_tensor * c_t;  // Cell state
};

struct lstm_state lstm_step(
    struct ggml_context * ctx,
    struct ggml_tensor  * x_t,
    struct ggml_tensor  * h_prev,
    struct ggml_tensor  * c_prev,
    struct ggml_tensor  * W_i, struct ggml_tensor * U_i, struct ggml_tensor * b_i,
    struct ggml_tensor  * W_f, struct ggml_tensor * U_f, struct ggml_tensor * b_f,
    struct ggml_tensor  * W_o, struct ggml_tensor * U_o, struct ggml_tensor * b_o,
    struct ggml_tensor  * W_g, struct ggml_tensor * U_g, struct ggml_tensor * b_g
) {
    // Input gate: i_t = sigmoid(W_i * x_t + U_i * h_prev + b_i)
    struct ggml_tensor * i_t = ggml_sigmoid(ctx, 
        ggml_add(ctx, 
            ggml_add(ctx, ggml_mul_mat(ctx, W_i, x_t), ggml_mul_mat(ctx, U_i, h_prev)),
            b_i
        )
    );

    // Forget gate: f_t = sigmoid(W_f * x_t + U_f * h_prev + b_f)
    struct ggml_tensor * f_t = ggml_sigmoid(ctx, 
        ggml_add(ctx, 
            ggml_add(ctx, ggml_mul_mat(ctx, W_f, x_t), ggml_mul_mat(ctx, U_f, h_prev)),
            b_f
        )
    );

    // Output gate: o_t = sigmoid(W_o * x_t + U_o * h_prev + b_o)
    struct ggml_tensor * o_t = ggml_sigmoid(ctx, 
        ggml_add(ctx, 
            ggml_add(ctx, ggml_mul_mat(ctx, W_o, x_t), ggml_mul_mat(ctx, U_o, h_prev)),
            b_o
        )
    );

    // Cell candidate: g_t = tanh(W_g * x_t + U_g * h_prev + b_g)
    struct ggml_tensor * g_t = ggml_tanh(ctx, 
        ggml_add(ctx, 
            ggml_add(ctx, ggml_mul_mat(ctx, W_g, x_t), ggml_mul_mat(ctx, U_g, h_prev)),
            b_g
        )
    );

    // Updated cell state: c_t = f_t * c_prev + i_t * g_t
    struct ggml_tensor * c_t = ggml_add(ctx,
        ggml_mul(ctx, f_t, c_prev),
        ggml_mul(ctx, i_t, g_t)
    );

    // Updated hidden state: h_t = o_t * tanh(c_t)
    struct ggml_tensor * h_t = ggml_mul(ctx, o_t, ggml_tanh(ctx, c_t));

    struct lstm_state state = { h_t, c_t };
    return state;
}