#pragma once
#include "Operation.h"

class GlobalAvgPool : public Operation {
  public:
    GlobalAvgPool(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    GlobalAvgPool(const GlobalAvgPool& src);
    virtual void initialize_tiles(MappingTable& mapping_table) override;

  protected:
    virtual void initialize_instructions(Tile* tile, Mapping mapping) override;
  private:
    std::vector<uint32_t> _kernel_shape;
    std::vector<uint32_t> _strides;
    // std::vector<uint32_t> _dilations;
    // std::vector<uint32_t> _pads;  
};