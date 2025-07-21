#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <cassert>
#include <iostream>
#include <utility>
#include "ggml.h"
#include "utils.h"
#include "encoder.h"

using namespace encodec;

class RandomTensorFactory {
public:
    explicit RandomTensorFactory(ggml_context* ctx)
        : ctx_{ctx},
          rng_{static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count())},
          dist_{-1.f, 1.f} {}

    Tensor* tensor_1d(int64_t len) {
        auto* t = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, len);
        fill(reinterpret_cast<float*>(t->data), len);
        return t;
    }

    Tensor* tensor_2d(int64_t rows, int64_t cols) { // note: ggml order is (cols, rows)
        auto* t = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, cols, rows);
        fill(reinterpret_cast<float*>(t->data), rows * cols);
        return t;
    }

    Tensor* tensor_3d(int64_t depth, int64_t rows, int64_t cols) { // ggml order is (cols, rows, depth)
        auto* t = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, cols, rows, depth);
        fill(reinterpret_cast<float*>(t->data), depth * rows * cols);
        return t;
    }

private:
    void fill(float* data, int64_t n) {
        for (int64_t i = 0; i < n; ++i) data[i] = dist_(rng_);
    }

    ggml_context* ctx_;
    std::mt19937                            rng_;
    std::uniform_real_distribution<float>   dist_;
};

struct RandomModelConfig {
    int C              = 4;
    int ks             = 3;
    int num_resnet     = 2;
    int in_ch          = 4;
    int bottleneck_ch  = 2;
    int out_ch         = 4;
    int down_blocks    = 2;
    int num_stages     = 2;
    int codebook_size  = 8;
    int input_len      = 16;
    int input_channels = 4;
};

struct RandomModel {
    Weights weights;
    Tensor* input;
};

inline RandomModel make_random_model(ggml_context* ctx, const RandomModelConfig& cfg = {}) {
    RandomTensorFactory rnd{ctx};
    RandomModel model;

    // First conv
    model.weights.first_conv = {
        rnd.tensor_1d(cfg.C),
        rnd.tensor_3d(cfg.C, cfg.input_channels, cfg.ks),
        rnd.tensor_1d(cfg.C)
    };

    // ResNet blocks
    model.weights.resnet_blocks.resize(cfg.num_resnet);
    for (int i = 0; i < cfg.num_resnet; ++i) {
        model.weights.resnet_blocks[i] = {
            /* bottleneck */ {rnd.tensor_1d(cfg.bottleneck_ch),
                              rnd.tensor_3d(cfg.bottleneck_ch, cfg.in_ch, 3),
                              rnd.tensor_1d(cfg.bottleneck_ch)},
            /* conv1x1   */ {rnd.tensor_1d(cfg.out_ch),
                              rnd.tensor_3d(cfg.out_ch, cfg.bottleneck_ch, 1),
                              rnd.tensor_1d(cfg.out_ch)}
        };
    }

    // Down‑sampling
    model.weights.downsample.resize(cfg.down_blocks);
    for (int i = 0; i < cfg.down_blocks; ++i) {
        model.weights.downsample[i] = {
            rnd.tensor_1d(cfg.out_ch),
            rnd.tensor_3d(cfg.out_ch, cfg.in_ch, cfg.ks),
            rnd.tensor_1d(cfg.out_ch)
        };
    }

    // LSTM
    const int H = cfg.out_ch;
    const int D = cfg.in_ch;
    model.weights.lstm = {
        rnd.tensor_2d(4 * H, D),
        rnd.tensor_2d(4 * H, H),
        rnd.tensor_1d(4 * H),
        rnd.tensor_1d(4 * H)
    };

    // Codebooks
    model.weights.codebooks.resize(cfg.num_stages);
    for (int i = 0; i < cfg.num_stages; ++i) {
        model.weights.codebooks[i] = { rnd.tensor_2d(cfg.codebook_size, cfg.out_ch) };
    }

    // Random input (B=1)
    model.input = rnd.tensor_3d(1, cfg.input_channels, cfg.input_len);

    return model;
}


int main() {
    ggml_init_params params{
        .mem_size   = 512 * 1024 * 1024ULL, // 512 MiB
        .mem_buffer = nullptr,
        .no_alloc   = false
    };
    ggml_context* ctx = ggml_init(params);

    RandomModelConfig cfg;
    auto model = make_random_model(ctx, cfg);

    encodec::Encoder encoder{ctx, std::move(model.weights)};
    auto* codes = encoder(model.input, /*threads*/1);

    print_ggml_3d_tensor(codes);

    ggml_free(ctx);
    return 0;
}
