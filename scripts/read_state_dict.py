import torch

state_dict = torch.load('model_dicts/compression_state_dict.bin', map_location='cpu')

print(state_dict['best_state'].keys())


"""
GGUF Model Format Specification

1. Header
   - Magic number: 4 bytes — 0x47 0x47 0x55 0x46 ("GGUF")
   - Format ID: 4 bytes (uint32) — must match 'ggml'

2. Model Hyperparameters
   - GGUF version: 4 bytes (uint32), currently 3
   - Tensor count: 8 bytes (uint64)
   - Metadata KV count: 8 bytes (uint64)

3. Metadata Section
   - Repeats 'KV count' times:
     - Key-value pairs, e.g.:
       in_channels, hidden_dim, n_filters, kernel_size,
       residual_kernel_size, n_bins, bandwidth, sr, ftype

4. Tensor Metadata Section
   - Repeats 'Tensor count' times:
     - name: GGUF string
     - n_dimensions: uint32
     - dimensions: uint64[]
     - type: uint32 (0 = float32, 1 = float16, etc.)
     - offset: uint64 (file position of tensor data)

5. Tensor Data
   - Raw tensor bytes begin at specified offsets and continue to EOF.
"""