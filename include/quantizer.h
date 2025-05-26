#pragma once

#include <cassert>
#include <vector>
#include <cstdio>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "utils.h"

// One quantization block: holds a K×D embedding table.
struct quant_block {
    struct ggml_tensor *embed;  // [K, D]
};

// A stack of quantization blocks (one per stage).
struct quantizer {
    std::vector<quant_block> blocks;
};

// Encode: map continuous vectors → discrete codes.
//   encoded_inp: [seq_length, D]
// returns codes: [seq_length, n_q]
static struct ggml_tensor *quantizer_encode(
    const struct quantizer *quant, struct ggml_context *ctx,
    struct ggml_tensor *encoded_inp)
{
    if (!encoded_inp) {
        std::fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const int seq_length = encoded_inp->ne[0];
    const int n_q        = (int)quant->blocks.size();

    // codes[i,j] = code index for time i, quantizer j
    struct ggml_tensor *codes = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, seq_length, n_q);
    ggml_set_input(codes);

    // transpose so that residual is [D×seq_length]
    struct ggml_tensor *inpL     = ggml_cont(ctx, ggml_transpose(ctx, encoded_inp));
    struct ggml_tensor *residual = inpL;
    struct ggml_tensor *indices;

    for (int i = 0; i < n_q; ++i) {
        const quant_block block = quant->blocks[i];

        // compute -2 * (embed × residual)  →  [K×seq_length]
        struct ggml_tensor *dp = ggml_scale(
            ctx,
            ggml_mul_mat(ctx, block.embed, residual),
            -2.0f
        );

        // precompute norms
        // embed norms: [K]     (sum of squares per row)
        struct ggml_tensor *sqr_embed     = ggml_sqr(ctx, block.embed);
        struct ggml_tensor *sqr_embed_nrm = ggml_sum_rows(ctx, sqr_embed);

        // input norms: [seq_length]  (sum of squares per column of residual)
        struct ggml_tensor *sqr_inp     = ggml_sqr(ctx, residual);
        struct ggml_tensor *sqr_inp_nrm = ggml_sum_rows(ctx, sqr_inp);

        // build full distance matrix:
        // dist = - (||x||^2 + ||e||^2 - 2 x·e)
        //    = -||x||^2 + (-2 x·e) - ||e||^2
        struct ggml_tensor *dist = ggml_add(
            ctx,
            ggml_repeat(ctx, sqr_inp_nrm,   dp),  // [K×seq_length]
            dp
        );
        dist = ggml_add(
            ctx,
            ggml_repeat(ctx, ggml_transpose(ctx, sqr_embed_nrm), dist),
            dist
        );
        dist = ggml_neg(ctx, dist);

        // 4) pick best code per time step: [seq_length]
        indices = ggml_argmax(ctx, dist);

        // 5) lookup embeddings [seq_length, D]
        struct ggml_tensor *quantized = ggml_get_rows(ctx, block.embed, indices);

        // 6) update residual
        residual = ggml_sub(ctx, residual, quantized);

        // 7) write indices into codes[:, i]
        codes = ggml_set_1d(
            ctx, codes, indices,
            /*offset=*/ i * codes->nb[1]
        );
    }

    return codes;
}

// Decode: map discrete codes → reconstructed vectors.
//   codes: [seq_length, n_q]
// returns quantized_out: [seq_length, D]
static struct ggml_tensor *quantizer_decode(
    const struct quantizer *quant, struct ggml_context *ctx,
    struct ggml_tensor *codes)
{
    if (!codes) {
        std::fprintf(stderr, "%s: null codes tensor\n", __func__);
        return NULL;
    }

    const int seq_length = codes->ne[0];
    const int n_q        = codes->ne[1];
    assert(n_q == (int)quant->blocks.size());

    // hidden_dim = D
    const int hidden_dim = quant->blocks[0].embed->ne[1];

    // accumulate quantized output in [D×seq_length]
    struct ggml_tensor *quantized_out = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hidden_dim, seq_length);
    ggml_set_input(quantized_out);

    for (int i = 0; i < n_q; ++i) {
        // view column i of codes: [seq_length]
        struct ggml_tensor *indices = ggml_view_1d(
            ctx, codes, seq_length, i * codes->nb[1]
        );

        // lookup embeddings → [seq_length, D]
        struct ggml_tensor *quantized = ggml_get_rows(
            ctx, quant->blocks[i].embed, indices
        );

        // accumulate
        quantized_out = ggml_add(ctx, quantized_out, quantized);
    }

    // transpose back → [seq_length, D]
    quantized_out = ggml_cont(ctx, ggml_transpose(ctx, quantized_out));
    return quantized_out;
}
