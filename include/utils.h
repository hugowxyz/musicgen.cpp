#pragma once

#include "ggml-cpu.h"
#include "ggml.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_ggml_1d_tensor(const ggml_tensor *tensor) {
    const int64_t ne0 = tensor->ne[0];
    float *data = (float *)tensor->data;

    printf("Result (shape %lld):\n[", ne0);
    for (int64_t i = 0; i < ne0; ++i) {
        printf(" %.2f", data[i]);
    }
    printf(" ]\n");
}

void print_ggml_2d_tensor(const ggml_tensor *tensor) {
    const int64_t ne0 = tensor->ne[0]; // columns (innermost)
    const int64_t ne1 = tensor->ne[1]; // rows
    float *data = (float *)tensor->data;

    printf("Result (shape %lld x %lld):\n[", ne0, ne1);
    for (int64_t j = 0; j < ne1; ++j) {
        if (j > 0)
            printf("\n");
        for (int64_t i = 0; i < ne0; ++i) {
            int64_t idx = i + j * ne0;
            printf(" %.2f", data[idx]);
        }
    }
    printf(" ]\n");
}

void print_ggml_3d_tensor(const struct ggml_tensor *tensor) {
    const int64_t ne0 = tensor->ne[0]; // innermost
    const int64_t ne1 = tensor->ne[1]; // middle
    const int64_t ne2 = tensor->ne[2]; // outermost

    printf("Result (shape %lld x %lld x %lld):\n", ne0, ne1, ne2);

    for (int64_t k = 0; k < ne2; ++k) {
        printf("Slice %lld:\n[", k);
        for (int64_t j = 0; j < ne1; ++j) {
            if (j > 0)
                printf("\n");
            for (int64_t i = 0; i < ne0; ++i) {
                int64_t idx = i + j * ne0 + k * ne0 * ne1;
                float val;
                if (tensor->type == GGML_TYPE_F16) {
                    ggml_fp16_t *data_f16 = (ggml_fp16_t *)tensor->data;
                    val = ggml_fp16_to_fp32(data_f16[idx]);
                } else if (tensor->type == GGML_TYPE_F32) {
                    float *data_f32 = (float *)tensor->data;
                    val = data_f32[idx];
                } else {
                    printf("Unsupported tensor type\n");
                    return;
                }
                printf(" %.2f", val);
            }
        }
        printf(" ]\n\n");
    }
}


struct ggml_tensor *create_1d_tensor(struct ggml_context *ctx, const float *data, int64_t ne0) {
    struct ggml_tensor *tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    memcpy(tensor->data, data, ne0 * sizeof(float));
    return tensor;
}

struct ggml_tensor *create_2d_tensor(struct ggml_context *ctx, const float *data, int64_t ne1, int64_t ne0) {
    // ne1 = rows, ne0 = cols
    struct ggml_tensor *tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    memcpy(tensor->data, data, ne0 * ne1 * sizeof(float));
    return tensor;
}

struct ggml_tensor *create_3d_tensor(struct ggml_context *ctx, const float *data, int64_t ne2, int64_t ne1, int64_t ne0) {
    // ne2 = depth, ne1 = rows, ne0 = cols
    struct ggml_tensor *tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    memcpy(tensor->data, data, ne0 * ne1 * ne2 * sizeof(float));
    return tensor;
}

struct ggml_tensor * compute_graph_from_tensor(struct ggml_context * ctx, struct ggml_tensor * final_tensor, int n_threads) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, final_tensor);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    return ggml_graph_node(gf, -1);
}
