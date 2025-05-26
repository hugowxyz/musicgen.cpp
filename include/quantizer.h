#pragma once

#include "ggml.h"

struct ggml_tensor * ggml_euclidean_distances(struct ggml_context * ctx,
                                              struct ggml_tensor * residual,     // [B, D]
                                              struct ggml_tensor * codebook)     // [K, D]
{
    const int64_t B = residual->ne[1];  // batch_size
    const int64_t D = residual->ne[0];  // dim
    const int64_t K = codebook->ne[1];  // codebook_size

    // Reshape to [D, 1, B] and [D, K, 1] for broadcasting
    struct ggml_tensor * residual_exp = ggml_reshape_3d(ctx, residual, D, 1, B); // [D,1,B]
    struct ggml_tensor * codebook_exp = ggml_reshape_3d(ctx, codebook, D, K, 1);  // [D,K,1]

    // Subtract: [D, K, B]
    struct ggml_tensor * diff = ggml_sub(ctx, residual_exp, codebook_exp);

    // Square differences: [D, K, B]
    struct ggml_tensor * sqr = ggml_mul(ctx, diff, diff);

    // Reshape to collapse D: [D, K * B]
    struct ggml_tensor * sqr_flat = ggml_reshape_2d(ctx, sqr, D, K * B);

    // Sum across D (first dimension), result [1, K * B]
    struct ggml_tensor * sum = ggml_sum_rows(ctx, sqr_flat);

    // Reshape to [K, B]
    struct ggml_tensor * sum_reshaped = ggml_reshape_2d(ctx, sum, K, B);

    // Transpose to [B, K]
    struct ggml_tensor * sum_T = ggml_transpose(ctx, sum_reshaped);

    // Optional square root to get actual Euclidean distance
    struct ggml_tensor * dist = ggml_sqrt(ctx, sum_T);

    return dist;  // [B, K]
}

struct ggml_tensor * rvq_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * input,                // [batch_size, dim]
    struct ggml_tensor ** codebooks,            // array of [codebook_size, dim] for each stage
    int                  num_stages,
    int                  codebook_size
) {
    struct ggml_tensor * residual = input;
    struct ggml_tensor * reconstruction = ggml_new_tensor_2d(ctx, input->type, input->ne[0], input->ne[1]);


    for (int k = 0; k < num_stages; ++k) {
        struct ggml_tensor * codebook = codebooks[k];  // [codebook_size, dim]

        // Compute Euclidean distances between residual and all codebook entries
        struct ggml_tensor * distances = ggml_euclidean_distances(ctx, residual, codebook); // [batch_size, codebook_size]

        // Find nearest codebook index
        struct ggml_tensor * neg_distances = ggml_neg(ctx, distances);
        struct ggml_tensor * nearest_index = ggml_argmax(ctx, neg_distances);  // [batch_size]

        // Gather nearest codebook vectors
        struct ggml_tensor * q_k = ggml_get_rows(ctx, codebook, nearest_index); // [batch_size, dim]

        // Accumulate quantized vector to reconstruction
        reconstruction = ggml_add(ctx, reconstruction, q_k);

        // Update residual
        residual = ggml_sub(ctx, residual, q_k);
    }

    return reconstruction;
}
