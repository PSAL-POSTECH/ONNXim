#pragma once
#include "Operation.h"

class Softmax : public Operation {
public:
    Softmax(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    Softmax(SimulationConfig config, MappingTable& mapping_table,
       std::vector<uint32_t> input_shape, uint32_t target_core=0);
    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _output_shape;

    uint32_t _seq;
    uint32_t _dk;
    uint32_t _tokens_per_tile;

    void calculate_loops();
    void initialize_tiles(MappingTable& mapping_table) override;
    void initialize_instructions(Tile* tile, Mapping mapping, uint32_t token_offset, uint32_t tokens);
};