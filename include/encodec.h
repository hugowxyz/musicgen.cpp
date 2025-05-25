#pragma once

#include "ggml.h"
#include <string>
#include <map>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstring>

// Model hyperparameters structure
struct encodec_hparams {
    int32_t n_vocab;
    int32_t n_audio_ctx;
    int32_t n_embd;
    int32_t n_head;
    int32_t n_layer;
    int32_t ftype;
};

// Model structure
struct encodec_model {
    encodec_hparams hparams;
    
    // model tensors
    struct ggml_tensor * encoder_conv1_w;
    struct ggml_tensor * encoder_conv1_b;
    // Add more tensors as needed
    
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Function to load model weights
bool encodec_model_load(const std::string & fname, encodec_model & model) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // Verify magic number
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // Load hyperparameters
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab,     sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_audio_ctx, sizeof(hparams.n_audio_ctx));
        fin.read((char *) &hparams.n_embd,      sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,      sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer,     sizeof(hparams.n_layer));
        fin.read((char *) &hparams.ftype,       sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_vocab     = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_audio_ctx = %d\n", __func__, hparams.n_audio_ctx);
        printf("%s: n_embd      = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head      = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer     = %d\n", __func__, hparams.n_layer);
        printf("%s: ftype       = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr       = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // Create context
    {
        const auto & hparams = model.hparams;
        
        // Calculate size needed for weights
        const int n_tensors = 2; // Adjust based on your model's needs
        size_t ctx_size = 1024*1024*1024; // 1 GB - adjust based on your needs
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // Prepare tensors
    {
        const auto & hparams = model.hparams;

        model.encoder_conv1_w = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, hparams.n_embd, 1);
        model.encoder_conv1_b = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, hparams.n_embd);

        // Map tensors to names
        model.tensors["encoder.conv1.weight"] = model.encoder_conv1_w;
        model.tensors["encoder.conv1.bias"] = model.encoder_conv1_b;
    }

    // Load weights
    {
        size_t total_size = 0;
        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = model.tensors[name];
            ggml_set_name(tensor, name.c_str());
            
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.c_str(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            read_buf.resize(ggml_nbytes(tensor));
            fin.read(read_buf.data(), ggml_nbytes(tensor));
            memcpy(tensor->data, read_buf.data(), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
        }

        printf("%s: model size = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return true;
}