#pragma once

#include <cassert>
#include <vector>
#include <cstdio>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "utils.h"

// One quantization block containing a codebook embedding matrix.
// - embed: [K, D] where:
//     K = codebook size (number of code vectors)
//     D = embedding dimensionality (e.g., hidden_dim)
struct quant_block {
    struct ggml_tensor *embed;  // [K, D]
};

// A vector quantizer composed of n_q stages (i.e., multiple quant_blocks).
// - blocks.size() = n_q
struct quantizer {
    std::vector<quant_block> blocks;
};

// Encode: maps continuous input vectors to discrete token indices.
//
// Parameters:
// - quant: quantizer containing n_q codebooks (each [K, D])
// - ctx:   ggml computation context
// - encoded_inp: input tensor of shape [seq_length, D]
//     where:
//       seq_length = number of time steps
//       D          = embedding dimension (must match codebook D)
//
// Returns:
// - codes: output tensor of shape [seq_length, n_q]
//     Each row contains `n_q` code indices (one per quantization stage).
static struct ggml_tensor *quantizer_encode(
    const struct quantizer *quant, struct ggml_context *ctx,
    struct ggml_tensor *encoded_inp)
{
    if (!encoded_inp) {
        std::fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const int seq_length = encoded_inp->ne[1];      // input rows
    const int n_q        = (int)quant->blocks.size(); // num quantizer stages

    // Output tensor: [seq_length, n_q] of int32 token indices
    struct ggml_tensor *codes = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, seq_length, n_q);
    ggml_set_input(codes);

    // Transpose input to [D, seq_length] for matmul compatibility
    // struct ggml_tensor *inpL     = ggml_cont(ctx, ggml_transpose(ctx, encoded_inp));
    struct ggml_tensor *inpL     = encoded_inp;
    struct ggml_tensor *residual = inpL;
    struct ggml_tensor *indices;

    for (int i = 0; i < n_q; ++i) {
        const quant_block block = quant->blocks[i];
        
        // auto block_embed = ggml_cont(ctx, ggml_transpose(ctx, block.embed));

        // Compute inner product: -2 * (embed × residual)  → [K, seq_length]
        struct ggml_tensor *dp = ggml_scale(
            ctx,
            ggml_mul_mat(ctx, block.embed, residual),
            -2.0f
        );

        // Precompute squared norms
        // Codebook norms: [K]
        struct ggml_tensor *sqr_embed     = ggml_sqr(ctx, block.embed);
        struct ggml_tensor *sqr_embed_nrm = ggml_sum_rows(ctx, sqr_embed);

        // Residual norms: [seq_length]
        struct ggml_tensor *sqr_inp     = ggml_sqr(ctx, residual);
        struct ggml_tensor *sqr_inp_nrm = ggml_sum_rows(ctx, sqr_inp);

        // Compute pairwise distances using:
        //   dist(x, e) = ||x - e||^2 = ||x||^2 + ||e||^2 - 2 * x·e
        struct ggml_tensor *dist = ggml_add(
            ctx,
            ggml_repeat(ctx, sqr_inp_nrm, dp),  // broadcast [seq_length] → [K, seq_length]
            dp
        );
        dist = ggml_add(
            ctx,
            ggml_repeat(ctx, ggml_transpose(ctx, sqr_embed_nrm), dist),
            dist
        );
        dist = ggml_neg(ctx, dist); // negate to allow argmax as min search

        // Select closest code: [seq_length]
        indices = ggml_argmax(ctx, dist);

        // Look up embedding vectors for selected codes: [seq_length, D]
        struct ggml_tensor *quantized = ggml_get_rows(ctx, block.embed, indices);

        // Update residual: residual = residual - quantized
        residual = ggml_sub(ctx, residual, quantized);

        // Write indices to codes[:, i]
        codes = ggml_set_1d(
            ctx, codes, indices,
            /*offset=*/ i * codes->nb[1]
        );
    }

    return codes;
}

// Decode: maps discrete code indices back to quantized embeddings.
//
// Parameters:
// - quant: quantizer with n_q blocks (same as used for encoding)
// - ctx:   ggml computation context
// - codes: [seq_length, n_q] int32 tensor of code indices
//
// Returns:
// - quantized_out: [seq_length, D] reconstructed embedding vectors
static struct ggml_tensor *quantizer_decode(
    const struct quantizer *quant, struct ggml_context *ctx,
    struct ggml_tensor *codes)
{
    if (!codes) {
        std::fprintf(stderr, "%s: null codes tensor\n", __func__);
        return NULL;
    }

    const int seq_length = codes->ne[0];   // number of time steps
    const int n_q        = codes->ne[1];   // number of quantization stages
    assert(n_q == (int)quant->blocks.size());

    // Hidden dimension D from codebook
    const int hidden_dim = quant->blocks[0].embed->ne[0];

    struct ggml_tensor *quantized_sum = NULL;

    for (int i = 0; i < n_q; ++i) {
        // Extract indices for current stage: [seq_length]
        struct ggml_tensor *indices = ggml_view_1d(
            ctx, codes, seq_length, i * codes->nb[1]);
        
        // Lookup embeddings: [seq_length, D]
        struct ggml_tensor *quantized = ggml_get_rows(
            ctx, quant->blocks[i].embed, indices);

        // Accumulate embeddings
        if (quantized_sum == NULL) {
            quantized_sum = quantized;
        } else {
            quantized_sum = ggml_add(ctx, quantized_sum, quantized);
        }
    }

    // Transpose to [seq_length, D]
    return ggml_cont(ctx, quantized_sum);
}
