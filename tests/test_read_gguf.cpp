#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
    std::vector<float> data;
};

std::string read_string(std::ifstream &f) {
    uint32_t len;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string s(len, '\0');
    f.read(&s[0], len);
    return s;
}

int main() {
    std::ifstream f("model_dicts/compression_state_dict.gguf", std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    // GGUF magic and version
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "GGUF") {
        std::cerr << "Invalid GGUF magic\n";
        return 1;
    }

    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 3) {
        std::cerr << "Unsupported version\n";
        return 1;
    }

    uint64_t num_tensors, num_metadata;
    f.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
    f.read(reinterpret_cast<char*>(&num_metadata), sizeof(num_metadata));

    std::unordered_map<std::string, std::string> metadata;
    for (uint64_t i = 0; i < num_metadata; ++i) {
        std::string key = read_string(f);
        std::string value = read_string(f);
        metadata[key] = value;
    }

    struct TensorMeta {
        std::string name;
        std::vector<uint64_t> shape;
        uint64_t offset;
    };

    std::vector<TensorMeta> tensor_metas;
    for (uint64_t i = 0; i < num_tensors; ++i) {
        TensorMeta meta;
        meta.name = read_string(f);

        uint32_t ndim;
        f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        meta.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            f.read(reinterpret_cast<char*>(&meta.shape[d]), sizeof(uint64_t));
        }

        uint32_t dtype;
        f.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
        if (dtype != 0) {
            std::cerr << "Unsupported dtype (only float32 supported)\n";
            return 1;
        }

        f.read(reinterpret_cast<char*>(&meta.offset), sizeof(meta.offset));
        tensor_metas.push_back(meta);
    }

    std::vector<Tensor> tensors;
    for (const auto& meta : tensor_metas) {
        Tensor tensor;
        tensor.name = meta.name;
        tensor.shape = meta.shape;

        uint64_t numel = 1;
        for (auto d : meta.shape) numel *= d;

        tensor.data.resize(numel);
        f.seekg(meta.offset, std::ios::beg);
        f.read(reinterpret_cast<char*>(tensor.data.data()), numel * sizeof(float));

        tensors.push_back(std::move(tensor));
    }

    std::cout << "Parsed " << tensors.size() << " tensors and " 
              << metadata.size() << " metadata entries.\n";

    // Optional: print summary
    for (const auto& tensor : tensors) {
        std::cout << "Tensor: " << tensor.name << ", shape: [";
        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            std::cout << tensor.shape[i] << (i + 1 < tensor.shape.size() ? ", " : "");
        }
        std::cout << "]\n";
    }

    return 0;
}
