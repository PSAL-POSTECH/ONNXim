#pragma once

#include "Operation.h"

class MaxPool : public Operation {
  public:
    MaxPool(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    MaxPool(const MaxPool& src);
    virtual void initialize_tiles(MappingTable& mapping_table) override;
    virtual void initialize_instructions(Tile* tile, Mapping mapping) override;

  protected:
    // virtual void initialize_instructions(SimulationConfig config, Tile& tile) override;
  private:
    std::vector<uint32_t> _kernel_shape;
    std::vector<uint32_t> _strides;
    std::vector<uint32_t> _dilations;
    std::vector<uint32_t> _pads;  
};