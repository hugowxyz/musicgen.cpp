#include "seanet.h"
#include "utils.h"

#include <cstdlib>
#include <ctime>
#include <cstring>
#include <vector>

void test_seanet_resnet_block() {
    const size_t ctx_size = 5 * 50000;
    void *ctx_data = malloc(ctx_size);
    struct ggml_context *ctx =
        ggml_init({.mem_size = ctx_size, .mem_buffer = ctx_data});

    const int64_t in_ch = 64;
    const int64_t bottleneck_ch = 32;
    const int64_t out_ch = 64;
    const int64_t B = 1;  // batch
    const int64_t T = 16; // time steps

    std::srand(std::time(nullptr));

    // allocate and fill weight & bias vectors
    std::vector<float> weight_g1_data(bottleneck_ch * 1 * 1), bias1_data(bottleneck_ch);
    std::vector<float> weight_v1_data(bottleneck_ch * in_ch * 3);
    std::vector<float> weight_g2_data(out_ch * 1 * 1), bias2_data(out_ch);
    std::vector<float> weight_v2_data(out_ch * bottleneck_ch * 1);

    auto fill_random = [](auto &vec) {
        for (auto &x : vec)
            x = (float(std::rand()) / RAND_MAX) * 2 - 1;
    };
    fill_random(weight_g1_data);
    fill_random(bias1_data);
    fill_random(weight_v1_data);
    fill_random(weight_g2_data);
    fill_random(bias2_data);
    fill_random(weight_v2_data);

    std::vector<float> input_data(B * in_ch * T);
    fill_random(input_data);

    // build ggml tensors
    auto *input = create_3d_tensor(ctx, input_data.data(), B, in_ch, T);

    // encoder.model.1.block.1.conv.conv:
    //   weight_g: [32,1,1], bias: [32], weight_v: [32,64,3]
    auto *g1 = create_3d_tensor(ctx, weight_g1_data.data(),
                                 bottleneck_ch, 1, 1);
    auto *b1 = create_1d_tensor(ctx, bias1_data.data(),
                                 bottleneck_ch);
    auto *v1 = create_3d_tensor(ctx, weight_v1_data.data(),
                                 bottleneck_ch, in_ch, 3);

    // encoder.model.1.block.3.conv.conv:
    //   weight_g: [64,1,1], bias: [64], weight_v: [64,32,1]
    auto *g2 = create_3d_tensor(ctx, weight_g2_data.data(),
                                 out_ch, 1, 1);
    auto *b2 = create_1d_tensor(ctx, bias2_data.data(),
                                 out_ch);
    auto *v2 = create_3d_tensor(ctx, weight_v2_data.data(),
                                 out_ch, bottleneck_ch, 1);

    auto *block_out = seanet_resnet_block(ctx, input, g1, v1, b1, g2, v2, b2);

    auto *result = compute_graph_from_tensor(ctx, block_out, /*n_threads=*/1);

    print_ggml_3d_tensor(result);

    ggml_free(ctx);
    free(ctx_data);
}

int main() {
    test_seanet_resnet_block();
    return 0;
}
