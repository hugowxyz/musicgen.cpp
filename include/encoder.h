#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "conv.h"
#include "seanet.h"
#include "lstm.h"
#include "quantizer.h"
#include "utils.h"


// Encoder parameters, including initial, SEANet block, and downsampling conv params
struct encodec_encoder_params {
    struct ggml_tensor * input;

    // Initial Conv1D
    struct ggml_tensor * first_conv_weight_g; // [C]
    struct ggml_tensor * first_conv_weight_v; // [ks, 1, C]
    struct ggml_tensor * first_conv_bias;     // [C] or NULL

    // SEANet ResNet blocks
    int num_resnet_blocks;
    struct ggml_tensor ** resnet_weight_g1;      // [bottleneck_ch]
    struct ggml_tensor ** resnet_weight_v1;      // [3, in_ch, bottleneck_ch]
    struct ggml_tensor ** resnet_bias_bottleneck;// [bottleneck_ch] or NULL
    struct ggml_tensor ** resnet_weight_g2;      // [out_ch]
    struct ggml_tensor ** resnet_weight_v2;      // [1, bottleneck_ch, out_ch]
    struct ggml_tensor ** resnet_bias_1x1;       // [out_ch] or NULL

    // Downsampling
    struct ggml_tensor ** down_weight_g;         // [out_ch]
    struct ggml_tensor ** down_weight_v;         // [ks, in_ch, out_ch]
    struct ggml_tensor ** down_bias;             // [out_ch] or NULL

    // --- the new stacked LSTM weights ---
    struct ggml_tensor * weight_ih_l0;  // [4H, D]
    struct ggml_tensor * weight_hh_l0;  // [4H, H]
    struct ggml_tensor * bias_ih_l0;    // [4H]
    struct ggml_tensor * bias_hh_l0;    // [4H]

    // RVQ parameters
    struct ggml_tensor ** codebooks;
    int num_stages;
    int codebook_size;
};

struct ggml_cgraph * build_encodec_encoder_graph(
    struct ggml_context * ctx,
    struct encodec_encoder_params * params
) {
    auto * gf = ggml_new_graph(ctx);
    auto * x  = params->input;

    // Initial Conv1D (wn)
    x = streamable_conv1d_wn(
        ctx, x,
        params->first_conv_weight_g,
        params->first_conv_weight_v,
        params->first_conv_bias,
        1, /*stride*/
        0,
        // params->first_conv_weight_v->ne[0]/2, /*padding*/
        1  /*dilation*/
    );

    // Residual + downsampling
    for (int i = 0; i < params->num_resnet_blocks; ++i) {
        x = seanet_resnet_block(
            ctx, x,
            params->resnet_weight_g1[i],
            params->resnet_weight_v1[i],
            params->resnet_bias_bottleneck[i],
            params->resnet_weight_g2[i],
            params->resnet_weight_v2[i],
            params->resnet_bias_1x1[i]
        );
        int ks  = params->down_weight_v[i]->ne[0];
        int pad = ks/2;
        x = streamable_conv1d_wn(
            ctx, x,
            params->down_weight_g[i],
            params->down_weight_v[i],
            params->down_bias[i],
            2, pad, 1
        );
    }

    // LSTM unroll with stacked weights
    const int seq_len = x->ne[2];
    struct ggml_tensor * h_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[1], 1);
    struct ggml_tensor * c_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[1], 1);

    for (int t = 0; t < seq_len; ++t) {
        auto * x_t = ggml_view_1d(ctx, x, x->ne[1], (size_t)t * x->ne[1]);
        auto   st  = lstm_step(
            ctx,
            x_t, h_t, c_t,
            params->weight_ih_l0,
            params->weight_hh_l0,
            params->bias_ih_l0,
            params->bias_hh_l0
        );
        h_t = st.h_t;
        c_t = st.c_t;
    }

    // RVQ
    quantizer tmp_quant;
    tmp_quant.blocks.reserve(params->num_stages);
    for (int i = 0; i < params->num_stages; ++i) {
        quant_block qb;
        qb.embed = params->codebooks[i];  // each codebook: [hidden_dim, codebook_size]
        tmp_quant.blocks.push_back(qb);
    }

    // h_t is [hidden_dim, 1]; transpose to [1, hidden_dim] so quantizer_encode expects [seq_length, D]
    // need to check this
    // struct ggml_tensor * h_t_row = ggml_cont(ctx, ggml_transpose(ctx, h_t));
    struct ggml_tensor * out = quantizer_encode(
        &tmp_quant,
        ctx,
        h_t
    );


    ggml_build_forward_expand(gf, out);
    return gf;
}


struct ggml_tensor * compute_encodec_encoder(
    struct ggml_context * ctx,
    struct encodec_encoder_params * params,
    int n_threads
) {
    struct ggml_cgraph * gf = build_encodec_encoder_graph(ctx, params);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    return ggml_graph_node(gf, -1);
}

// int main() {
//     // // Allocate GGML context with some memory budget
//     // size_t ctx_size = 16 * 1024 * 1024; // 16MB, adjust as needed
//     // struct ggml_init_params init_params = {
//     //     .mem_size   = ctx_size,
//     //     .mem_buffer = malloc(ctx_size),
//     //     .no_alloc   = false
//     // };
//     // struct ggml_context * ctx = ggml_init(init_params);

//     // // Example dummy input: [B, 1, T]
//     // int batch_size = 1;
//     // int channels   = 1;
//     // int seq_len    = 256;
//     // struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, batch_size, seq_len);

//     // // Example dummy conv layers
//     // int num_resnet_blocks = 2;
//     // struct ggml_tensor * conv_weights[2];
//     // struct ggml_tensor * conv_biases[2];
//     // for (int i = 0; i < num_resnet_blocks; ++i) {
//     //     conv_weights[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 1, 3); // Example shape
//     //     conv_biases[i]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
//     // }

//     // // Example dummy resnet block weights
//     // struct ggml_tensor * resnet_weights_3x1[2];
//     // struct ggml_tensor * resnet_biases_3x1[2];
//     // struct ggml_tensor * resnet_weights_1x1[2];
//     // struct ggml_tensor * resnet_biases_1x1[2];
//     // for (int i = 0; i < num_resnet_blocks; ++i) {
//     //     resnet_weights_3x1[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 64, 3);
//     //     resnet_biases_3x1[i]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
//     //     resnet_weights_1x1[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 32, 1);
//     //     resnet_biases_1x1[i]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
//     // }

//     // // Example dummy LSTM weights (using small sizes for demonstration)
//     // int input_dim = 64;
//     // int hidden_dim = 64;
//     // struct ggml_tensor * W_i = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, hidden_dim);
//     // struct ggml_tensor * U_i = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, hidden_dim);
//     // struct ggml_tensor * b_i = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
//     // struct ggml_tensor * W_f = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, hidden_dim);
//     // struct ggml_tensor * U_f = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, hidden_dim);
//     // struct ggml_tensor * b_f = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
//     // struct ggml_tensor * W_o = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, hidden_dim);
//     // struct ggml_tensor * U_o = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, hidden_dim);
//     // struct ggml_tensor * b_o = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
//     // struct ggml_tensor * W_g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, hidden_dim);
//     // struct ggml_tensor * U_g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, hidden_dim);
//     // struct ggml_tensor * b_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);

//     // // Example dummy RVQ codebooks
//     // int num_stages = 4;
//     // int codebook_size = 256;
//     // struct ggml_tensor * codebooks[4];
//     // for (int i = 0; i < num_stages; ++i) {
//     //     codebooks[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, codebook_size);
//     // }

//     // // Fill encodec_encoder_params struct
//     // struct encodec_encoder_params params = {
//     //     .input = input,
//     //     .conv_weights = conv_weights,
//     //     .conv_biases = conv_biases,
//     //     .resnet_weights_3x1 = resnet_weights_3x1,
//     //     .resnet_biases_3x1 = resnet_biases_3x1,
//     //     .resnet_weights_1x1 = resnet_weights_1x1,
//     //     .resnet_biases_1x1 = resnet_biases_1x1,
//     //     .num_resnet_blocks = num_resnet_blocks,
//     //     .W_i = W_i, .U_i = U_i, .b_i = b_i,
//     //     .W_f = W_f, .U_f = U_f, .b_f = b_f,
//     //     .W_o = W_o, .U_o = U_o, .b_o = b_o,
//     //     .W_g = W_g, .U_g = U_g, .b_g = b_g,
//     //     .codebooks = codebooks,
//     //     .num_stages = num_stages,
//     //     .codebook_size = codebook_size
//     // };

//     // // Compute encoder output
//     // int n_threads = 4;
//     // struct ggml_tensor * output = compute_encodec_encoder(ctx, &params, n_threads);

//     // // Output dimensions
//     // printf("Encoder output shape: [%lld, %lld]\n", output->ne[1], output->ne[0]);

//     // ggml_free(ctx);
//     // free(init_params.mem_buffer);

//     return 0;
// }
