#include "ggml.h"

int main() {
    struct ggml_context * ctx = ggml_init(...);
    struct ggml_tensor * tensor = ggml_load_gguf(ctx, "model.gguf");

    
}
