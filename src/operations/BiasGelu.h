#pragma once
#include "Operation.h"

class BiasGelu : public Operation {
public:
    BiasGelu(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    BiasGelu(SimulationConfig config, Model* model, std::string name, std::map<std::string, std::string>& attributes, uint32_t target_core=0);

    std::vector<uint32_t> _bias_shape;

    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _output_shape;

    uint32_t _batch_size;
    uint32_t _seq;
    uint32_t _dk;
    uint32_t _tokens_per_tile;

    void calculate_loops();
    void initialize_tiles(MappingTable& mapping_table) override;
    void initialize_instructions(Tile* tile, Mapping mapping, uint32_t token_offset, uint32_t tokens);
};