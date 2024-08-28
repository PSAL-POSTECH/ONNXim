#pragma once

#include "Operation.h"

class AdaptiveAvgPool : public Operation {
 public:
  AdaptiveAvgPool(SimulationConfig config, Model* model,
                  onnx::NodeProto& node_proto, uint32_t target_core=0);
  AdaptiveAvgPool(const AdaptiveAvgPool& src);

  virtual void initialize_tiles(MappingTable& mapping_table) override;

 protected:
  virtual void initialize_instructions(Tile* tile, Mapping mapping);

 private:
  std::vector<uint32_t> _kernel_shape;
  std::vector<uint32_t> _strides;
  bool _skip = false;
};