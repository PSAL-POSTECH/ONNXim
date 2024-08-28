/*TODO: implement this */
#pragma once

#include "Operation.h"

class Concat : public Operation {
  public:
    Concat(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    Concat(const Concat& src);
    Concat(SimulationConfig config, Model* model, std::string name,
                  std::map<std::string, std::string>& attributes, uint32_t target_core=0);
    virtual void initialize_tiles(MappingTable& mapping_table) override;
    virtual void initialize_instructions(Tile* tile, Mapping mapping) override;
  protected:

  private:
    // std::vector<uint32_t> _kernel_shape;
    // std::vector<uint32_t> _strides;
    // std::vector<uint32_t> _dilations;
    // std::vector<uint32_t> _pads;

    uint32_t _axis;
};