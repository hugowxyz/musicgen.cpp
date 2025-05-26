#pragma once

#include <map>
#include <string>

#include "ggml.h"
#include "encoder.h"
#include "decoder.h"

struct encodec_model {
    encodec_encoder_params encoder_params;
    encodec_decoder_params decoder_params;

    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};