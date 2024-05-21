#pragma once
#include "Operation.h"

class BiasAct : public Operation {
public:
    BiasAct(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
    BiasAct(SimulationConfig config, Model* model, std::string name, std::map<std::string, std::string>& attributes);

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