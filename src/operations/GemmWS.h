#pragma once
#include "Gemm.h"

class GemmWS : public Gemm {
 public:
  GemmWS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
  GemmWS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, bool has_true);
  GemmWS(SimulationConfig config, MappingTable mapping_table,
         std::vector<uint32_t> input_shape, std::vector<uint32_t> weight_shape,
         std::vector<uint32_t> output_shape);
  virtual void initialize_tiles(MappingTable mapping_table) override;
  bool has_bias = true;
 protected:
  virtual void initialize_instructions(Tile& tile, Mapping mapping) override;
 private:
};