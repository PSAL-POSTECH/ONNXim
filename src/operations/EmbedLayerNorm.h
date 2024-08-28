#pragma once
//#include "../tensor/NPUTensor.h"
#include "Operation.h"

class EmbedLayerNorm: public Operation {
   public:
    EmbedLayerNorm(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);

    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _output_shape;
    std::vector<uint32_t> _weight_shape;
    std::vector<uint32_t> _position_weight_shape;
    std::vector<uint32_t> _token_type_weight;
    std::vector<uint32_t> _ln_weight_shape;
    std::vector<uint32_t> _ln_bias_shape;
    void initialize_tiles(MappingTable& mapping_table);
    void initialize_instructions(Tile* tile, Mapping mapping);
   protected:
};