#pragma once
//#include "../tensor/NPUTensor.h"
#include "Operation.h"
#include <numeric>

class Dummy: public Operation {
public:
   Dummy(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);

   std::vector<uint32_t> _input_shape;
   std::vector<uint32_t> _output_shape;
   void initialize_tiles(MappingTable& mapping_table);
   void initialize_instructions(Tile* tile, Mapping mapping);
   uint64_t _total_loop;
   uint32_t _element_in_tile;
};