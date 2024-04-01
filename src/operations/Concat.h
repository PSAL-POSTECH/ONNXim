/*TODO: implement this */
#pragma once

#include "Operation.h"

class Concat : public Operation {
  public:
    Concat(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
    Concat(const Concat& src);
    virtual void initialize_tiles(MappingTable& mapping_table) override;
    virtual void initialize_instructions(Tile* tile, Mapping mapping) override;
  protected:

  private:
    // std::vector<uint32_t> _kernel_shape;
    // std::vector<uint32_t> _strides;
    // std::vector<uint32_t> _dilations;
    // std::vector<uint32_t> _pads;

    int _axis;
};