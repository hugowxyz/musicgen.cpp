# EnCodec Model Architecture Inference

https://trello.com/b/GJ7FEAv0/musicgen-inference

## Architecture Overview

### Encoder

- Initial Convolution  
  - 1D Conv: (kernel_size, in_channels, n_filters)

- 4 Residual Downsampling Blocks  
  - Each block contains:
    - Conv1 (1D): (res_kernel_size, in_ch, out_ch / 2) — bottleneck
    - Conv2 (1D): (1, out_ch / 2, out_ch)
    <!-- - Shortcut Conv (1D): (1, in_ch, out_ch) Replaced by Identity-->
    - Downsampling Conv (1D): (2 * ratio, in_ch, 2 * out_ch)  
  - `mult` (channel multiplier) doubles after each block

- LSTM Layers  
  - Two layers (likely stacked or bidirectional)  
  - Weight/bias tensors suggest:  
    - input_dim = mult * n_filters  
    - hidden_dim = 4 * input_dim (typical for LSTM gates)

- Final Convolution  
  - 1D Conv: (kernel_size, input_ch, hidden_dim)

## Quantizer

- Vector Quantization  
  - n_q quantizer blocks  
  - Each has an embedding/codebook: (hidden_dim, n_bins)

## Decoder

- Initial Convolution  
  - 1D Conv: (kernel_size, hidden_dim, mult * n_filters)  
    - mult = 2^4 = 16 (inverse of encoder downscaling)

- LSTM Layers  
  - Same structure as encoder

- 4 Upsampling + Residual Blocks  
  - Each block contains:
    - Upsample Conv: (ratio * 2, in_ch / 2, in_ch)
    - Conv1: (res_kernel_size, in_ch / 2, in_ch / 4)
    - Conv2: (1, in_ch / 4, in_ch / 2)
    <!-- - Shortcut Conv: (1, in_ch / 2, in_ch / 2) Replaced by Identity -->
  - `mult` halves after each block

- Final Convolution  
  - 1D Conv: (kernel_size, n_filters, in_channels)
