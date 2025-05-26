// test_lstm.cpp

#include "ggml.h"
#include "lstm.h" // your LSTM implementation header
#include "utils.h" // create_{1d,2d}_tensor, compute_graph_from_tensor, print_ggml_1d_tensor

#include <cstdlib>
#include <ctime>
#include <vector>

void test_lstm_step() {
    // allocate GGML context
    const size_t ctx_size = 4 * 1024 * 1024;
    void *ctx_data = malloc(ctx_size);
    struct ggml_context *ctx =
        ggml_init({.mem_size = ctx_size, .mem_buffer = ctx_data});

    std::srand(std::time(nullptr));

    // dimensions
    const int D = 10;    // input size
    const int H = 20;    // hidden size
    const int G = 4 * H; // 4 gates

    // random buffers
    std::vector<float> x_data(D), h_prev_data(H), c_prev_data(H);
    std::vector<float> w_ih_data(G * D), w_hh_data(G * H);
    std::vector<float> b_ih_data(G), b_hh_data(G);

    auto fill_rand = [](std::vector<float> &v) {
        for (auto &x : v)
            x = (float(std::rand()) / RAND_MAX) * 2.f - 1.f;
    };
    fill_rand(x_data);
    fill_rand(h_prev_data);
    fill_rand(c_prev_data);
    fill_rand(w_ih_data);
    fill_rand(w_hh_data);
    fill_rand(b_ih_data);
    fill_rand(b_hh_data);

    // build tensors
    auto *x = create_1d_tensor(ctx, x_data.data(), D);
    auto *h_prev = create_1d_tensor(ctx, h_prev_data.data(), H);
    auto *c_prev = create_1d_tensor(ctx, c_prev_data.data(), H);
    auto *W_ih = create_2d_tensor(ctx, w_ih_data.data(), G, D);
    auto *W_hh = create_2d_tensor(ctx, w_hh_data.data(), G, H);
    auto *b_ih = create_1d_tensor(ctx, b_ih_data.data(), G);
    auto *b_hh = create_1d_tensor(ctx, b_hh_data.data(), G);

    // run one step
    struct lstm_state st =
        lstm_step(ctx, x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh);

    // execute graph for h_t and c_t
    auto *h_out = compute_graph_from_tensor(ctx, st.h_t, 1);
    auto *c_out = compute_graph_from_tensor(ctx, st.c_t, 1);

    // print results
    print_ggml_1d_tensor(h_out);
    print_ggml_1d_tensor(c_out);

    // cleanup
    ggml_free(ctx);
    free(ctx_data);
}

int main() {
    test_lstm_step();
    return 0;
}
