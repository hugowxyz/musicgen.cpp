#pragma once

#include "ggml.h"
#include "conv.h"
#include "seanet.h"
#include "lstm.h"
#include "quantizer.h"
#include "utils.h"

#include <vector>
#include <cassert>
#include <utility>

namespace encodec {

using Tensor = ggml_tensor;


struct Conv1dWeights {
    Tensor* g;      // g   ‑ [C]
    Tensor* v;      // v   ‑ [ks, in, out]
    Tensor* bias{}; // optional ‑ [C]
};

struct ResNetBlockWeights {
    Conv1dWeights bottleneck; // 3×1 conv (bottleneck)
    Conv1dWeights conv1x1;    // 1×1 conv (expansion)
};

// Down‑sampling uses the same weight layout as a normal conv
using DownsampleWeights = Conv1dWeights;

struct LSTMWeights {
    Tensor* weight_ih; // [4H, D]
    Tensor* weight_hh; // [4H, H]
    Tensor* bias_ih;   // [4H]
    Tensor* bias_hh;   // [4H]
};

struct QuantizerCodebook {
    Tensor* embed; // [hidden_dim, codebook_size]
};

//-------------------------------------
// Aggregate of all weights the encoder needs
//-------------------------------------
struct Weights {
    Conv1dWeights                   first_conv;
    std::vector<ResNetBlockWeights> resnet_blocks;
    std::vector<DownsampleWeights>  downsample;
    LSTMWeights                     lstm;
    std::vector<QuantizerCodebook>  codebooks;
};

//-------------------------------------
// The actual encoder
//-------------------------------------
class Encoder {
public:
    Encoder(ggml_context* ctx, Weights w) noexcept
        : ctx_{ctx}, w_{std::move(w)} {}

    /**
     * Encode a 3‑D input tensor (B, C=1, T).
     * @param input      Input tensor (ownership not taken).
     * @param n_threads  How many CPU threads to use.
     * @return           Pointer to the last node of the GGML graph (codes).
     */
    [[nodiscard]] Tensor* operator()(Tensor* input, int n_threads = 4) const {
        auto* graph = build_graph(input);
        ggml_graph_compute_with_ctx(ctx_, graph, n_threads);
        return ggml_graph_node(graph, -1);
    }

private:
    ggml_context* ctx_; // not owned
    Weights       w_;   // lightweight: just raw pointers

    ggml_cgraph* build_graph(Tensor* x) const {
        auto* gf = ggml_new_graph(ctx_);

        // Initial 1‑D conv (weight‑norm)
        x = streamable_conv1d_wn(ctx_, x,
                                 w_.first_conv.g,
                                 w_.first_conv.v,
                                 w_.first_conv.bias,
                                 /*stride*/1,
                                 /*pad*/0,
                                 /*dilation*/1);

        // ResNet + down‑sampling stages
        assert(w_.resnet_blocks.size() == w_.downsample.size());
        for (std::size_t i = 0; i < w_.resnet_blocks.size(); ++i) {
            const auto& res  = w_.resnet_blocks[i];
            const auto& down = w_.downsample[i];

            x = seanet_resnet_block(ctx_, x,
                                    res.bottleneck.g,
                                    res.bottleneck.v,
                                    res.bottleneck.bias,
                                    res.conv1x1.g,
                                    res.conv1x1.v,
                                    res.conv1x1.bias);

            const int ks  = down.v->ne[0];
            const int pad = ks / 2;
            x = streamable_conv1d_wn(ctx_, x,
                                     down.g,
                                     down.v,
                                     down.bias,
                                     /*stride*/2,
                                     pad,
                                     /*dilation*/1);
        }

        // --- LSTM unroll --------------------------------------
        const int seq_len = x->ne[2];
        Tensor* h_t = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, x->ne[1], 1);
        Tensor* c_t = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, x->ne[1], 1);

        for (int t = 0; t < seq_len; ++t) {
            Tensor* x_t = ggml_view_1d(ctx_, x, x->ne[1], static_cast<size_t>(t) * x->ne[1]);
            auto st = lstm_step(ctx_,
                                x_t, h_t, c_t,
                                w_.lstm.weight_ih,
                                w_.lstm.weight_hh,
                                w_.lstm.bias_ih,
                                w_.lstm.bias_hh);
            h_t = st.h_t;
            c_t = st.c_t;
        }

        // rvq
        quantizer q;
        q.blocks.reserve(w_.codebooks.size());
        for (const auto& cb : w_.codebooks) {
            q.blocks.push_back({cb.embed});
        }
        Tensor* out = quantizer_encode(&q, ctx_, h_t);

        ggml_build_forward_expand(gf, out);
        return gf;
    }
};

}
