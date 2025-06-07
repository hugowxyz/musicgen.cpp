#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include "ggml.h"
#include "utils.h"
#include "encoder.h"


// === Helpers ===

float *random_floats(int64_t size) {
    float *data = (float *)malloc(sizeof(float) * size);
    for (int64_t i = 0; i < size; i++)
        data[i] = ((float) rand() / RAND_MAX - 0.5f) * 2.0f;
    return data;
}

struct ggml_tensor *rand_1d(struct ggml_context *ctx, int64_t ne0) {
    float *data = random_floats(ne0);
    struct ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    memcpy(t->data, data, ne0 * sizeof(float));
    free(data);
    return t;
}

struct ggml_tensor *rand_2d(struct ggml_context *ctx, int64_t ne1, int64_t ne0) {
    float *data = random_floats(ne0 * ne1);
    struct ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    memcpy(t->data, data, ne0 * ne1 * sizeof(float));
    free(data);
    return t;
}

struct ggml_tensor *rand_3d(struct ggml_context *ctx, int64_t ne2, int64_t ne1, int64_t ne0) {
    float *data = random_floats(ne0 * ne1 * ne2);
    struct ggml_tensor *t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    memcpy(t->data, data, ne0 * ne1 * ne2 * sizeof(float));
    free(data);
    return t;
}

// === Initialization ===

void init_encoder_params(struct ggml_context *ctx, struct encodec_encoder_params *params,
                         int C, int ks,
                         int num_resnet_blocks, int in_ch, int bottleneck_ch, int out_ch,
                         int down_blocks, int num_stages, int codebook_size,
                         int input_len, int input_channels) {
    srand(time(NULL));

    params->first_conv_weight_g = rand_1d(ctx, C);
    snprintf(params->first_conv_weight_g->name, GGML_MAX_NAME, "first_conv_weight_g");

    params->first_conv_weight_v = rand_3d(ctx, C, input_channels, ks);  // [ks, in_ch=1, out_ch=C]
    snprintf(params->first_conv_weight_v->name, GGML_MAX_NAME, "first_conv_weight_v");

    params->first_conv_bias = rand_1d(ctx, C);
    snprintf(params->first_conv_bias->name, GGML_MAX_NAME, "first_conv_bias");

    params->num_resnet_blocks = num_resnet_blocks;
    params->resnet_weight_g1       = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * num_resnet_blocks);
    params->resnet_weight_v1       = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * num_resnet_blocks);
    params->resnet_bias_bottleneck = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * num_resnet_blocks);
    params->resnet_weight_g2       = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * num_resnet_blocks);
    params->resnet_weight_v2       = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * num_resnet_blocks);
    params->resnet_bias_1x1        = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * num_resnet_blocks);

    for (int i = 0; i < num_resnet_blocks; i++) {
        params->resnet_weight_g1[i] = rand_1d(ctx, bottleneck_ch);
        snprintf(params->resnet_weight_g1[i]->name, GGML_MAX_NAME, "resnet_%d_weight_g1", i);

        params->resnet_weight_v1[i] = rand_3d(ctx, bottleneck_ch, in_ch, 3);
        snprintf(params->resnet_weight_v1[i]->name, GGML_MAX_NAME, "resnet_%d_weight_v1", i);

        params->resnet_bias_bottleneck[i] = rand_1d(ctx, bottleneck_ch);
        snprintf(params->resnet_bias_bottleneck[i]->name, GGML_MAX_NAME, "resnet_%d_bias_bottleneck", i);

        params->resnet_weight_g2[i] = rand_1d(ctx, out_ch);
        snprintf(params->resnet_weight_g2[i]->name, GGML_MAX_NAME, "resnet_%d_weight_g2", i);

        params->resnet_weight_v2[i] = rand_3d(ctx, out_ch, bottleneck_ch, 1);
        snprintf(params->resnet_weight_v2[i]->name, GGML_MAX_NAME, "resnet_%d_weight_v2", i);

        params->resnet_bias_1x1[i] = rand_1d(ctx, out_ch);
        snprintf(params->resnet_bias_1x1[i]->name, GGML_MAX_NAME, "resnet_%d_bias_1x1", i);
    }

    params->down_weight_g = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * down_blocks);
    params->down_weight_v = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * down_blocks);
    params->down_bias     = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * down_blocks);

    for (int i = 0; i < down_blocks; i++) {
        params->down_weight_g[i] = rand_1d(ctx, out_ch);
        snprintf(params->down_weight_g[i]->name, GGML_MAX_NAME, "down_%d_weight_g", i);

        params->down_weight_v[i] = rand_3d(ctx, out_ch, in_ch, ks);
        snprintf(params->down_weight_v[i]->name, GGML_MAX_NAME, "down_%d_weight_v", i);

        params->down_bias[i] = rand_1d(ctx, out_ch);
        snprintf(params->down_bias[i]->name, GGML_MAX_NAME, "down_%d_bias", i);
    }

    int D = in_ch, H = out_ch;
    params->weight_ih_l0 = rand_2d(ctx, 4 * H, D);
    snprintf(params->weight_ih_l0->name, GGML_MAX_NAME, "weight_ih_l0");

    params->weight_hh_l0 = rand_2d(ctx, 4 * H, H);
    snprintf(params->weight_hh_l0->name, GGML_MAX_NAME, "weight_hh_l0");

    params->bias_ih_l0 = rand_1d(ctx, 4 * H);
    snprintf(params->bias_ih_l0->name, GGML_MAX_NAME, "bias_ih_l0");

    params->bias_hh_l0 = rand_1d(ctx, 4 * H);
    snprintf(params->bias_hh_l0->name, GGML_MAX_NAME, "bias_hh_l0");

    params->num_stages = num_stages;
    params->codebook_size = codebook_size;
    params->codebooks = (struct ggml_tensor **)malloc(sizeof(struct ggml_tensor *) * num_stages);
    for (int i = 0; i < num_stages; i++) {
        params->codebooks[i] = rand_2d(ctx, codebook_size, out_ch);
        snprintf(params->codebooks[i]->name, GGML_MAX_NAME, "codebook_%d", i);
    }

    params->input = rand_3d(ctx, 1, input_channels, input_len);
    snprintf(params->input->name, GGML_MAX_NAME, "input");
}


// === Example main ===

int main() {
    struct ggml_init_params ctx_params = {
        .mem_size   = 1024 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(ctx_params);

    struct encodec_encoder_params params;
    init_encoder_params(ctx, &params,
        /* C */ 4,
        /* ks */ 3,
        /* num_resnet_blocks */ 2,
        /* in_ch */ 4,
        /* bottleneck_ch */ 2,
        /* out_ch */ 4,
        /* down_blocks */ 2,
        /* num_stages */ 2,
        /* codebook_size */ 8,
        /* input_len */ 16,
        /* input_channels */ 4
    );

    printf("Initialization complete.\n");

    auto res = compute_encodec_encoder(ctx, &params, 1);

    print_ggml_3d_tensor(res);

    ggml_free(ctx);
    return 0;
}
