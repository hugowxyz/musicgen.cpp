import torch
import struct
import numpy as np

GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

state = torch.load('model_dicts/compression_state_dict.bin', map_location='cpu')
params = state['best_state']

metadata = {
    'kernel_size': 5
}

with open("model.gguf", "wb") as f:
    f.write(GGUF_MAGIC)
    f.write(struct.pack('<I', GGUF_VERSION))

    tensor_items = list(params.items())
    num_tensors = len(tensor_items)
    num_metadata = len(metadata)

    f.write(struct.pack('<Q', num_tensors)) 
    f.write(struct.pack('<Q', num_metadata)) 

    for k, v in metadata.items():
        k_enc = k.encode('utf-8')
        f.write(struct.pack('<I', len(k_enc)))
        f.write(k_enc)

        v_str = str(v).encode('utf-8')
        f.write(struct.pack('<I', len(v_str)))
        f.write(v_str)

    # Reserve tensor metadata
    tensor_data_offset = f.tell() + sum(
        4 + len(k.encode('utf-8')) + 4 + len(v.shape) * 8 + 4 + 8
        for k, v in params.items()
    )

    tensor_offsets = []
    for name, tensor in tensor_items:
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('<I', len(name_bytes)))
        f.write(name_bytes)

        # Dimensions
        shape = tensor.shape
        f.write(struct.pack('<I', len(shape)))
        for dim in shape:
            f.write(struct.pack('<Q', dim))

        # Data type (0 for float32)
        f.write(struct.pack('<I', 0))

        # Offset of tensor data
        f.write(struct.pack('<Q', tensor_data_offset))
        tensor_offsets.append(tensor_data_offset)

        # Update offset
        tensor_data_offset += tensor.numel() * 4  # float32 = 4 bytes


    for (_, tensor), offset in zip(tensor_items, tensor_offsets):
        tensor = tensor.to(torch.float32).contiguous()
        np_tensor = tensor.cpu().numpy().astype(np.float32)
        f.write(np_tensor.tobytes())
