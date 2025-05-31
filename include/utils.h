#pragma once

#include "ggml-cpu.h"
#include "ggml.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_scalar_value(const ggml_tensor *tensor, int64_t idx) {
    switch (tensor->type) {
        case GGML_TYPE_F32: {
            float *data = (float *)tensor->data;
            printf(" %.2f", data[idx]);
            break;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t *data = (ggml_fp16_t *)tensor->data;
            float val = ggml_fp16_to_fp32(data[idx]);
            printf(" %.2f", val);
            break;
        }
        case GGML_TYPE_I8: {
            int8_t *data = (int8_t *)tensor->data;
            printf(" %d", data[idx]);
            break;
        }
        case GGML_TYPE_I16: {
            int16_t *data = (int16_t *)tensor->data;
            printf(" %d", data[idx]);
            break;
        }
        case GGML_TYPE_I32: {
            int32_t *data = (int32_t *)tensor->data;
            printf(" %d", data[idx]);
            break;
        }
        default:
            printf(" (unsupported type %d)", tensor->type);
            break;
    }
}

void print_ggml_1d_tensor(const ggml_tensor *tensor) {
    const int64_t ne0 = tensor->ne[0];
    printf("Result (internal shape %lld):\n[", ne0);
    for (int64_t i = 0; i < ne0; ++i) {
        print_scalar_value(tensor, i);
    }
    printf(" ]\n");
}

void print_ggml_2d_tensor(const ggml_tensor *tensor) {
    const int64_t ne0 = tensor->ne[0]; // columns
    const int64_t ne1 = tensor->ne[1]; // rows

    printf("Result (internal shape %lld x %lld):\n[", ne0, ne1);
    for (int64_t j = 0; j < ne1; ++j) {
        if (j > 0) printf("\n ");
        for (int64_t i = 0; i < ne0; ++i) {
            int64_t idx = i + j * ne0;
            print_scalar_value(tensor, idx);
        }
    }
    printf(" ]\n");
}

void print_ggml_3d_tensor(const ggml_tensor *tensor) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];

    printf("Result (internal shape %lld x %lld x %lld):\n", ne0, ne1, ne2);
    for (int64_t k = 0; k < ne2; ++k) {
        printf("Slice %lld:\n[", k);
        for (int64_t j = 0; j < ne1; ++j) {
            if (j > 0) printf("\n ");
            for (int64_t i = 0; i < ne0; ++i) {
                int64_t idx = i + j * ne0 + k * ne0 * ne1;
                print_scalar_value(tensor, idx);
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
