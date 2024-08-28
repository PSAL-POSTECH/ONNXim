#pragma once
#include "Gemm.h"

class GemmWS : public Gemm {
 public:
  GemmWS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
  GemmWS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, bool has_true, uint32_t target_core=0);
  GemmWS(SimulationConfig config, MappingTable& mapping_table,
         std::vector<uint32_t> input_shape, std::vector<uint32_t> weight_shape,
         std::vector<uint32_t> output_shape, uint32_t target_core);
  GemmWS(SimulationConfig config, Model* model, std::string name, std::map<std::string, std::string>& attribute, uint32_t target_core);
  virtual void initialize_tiles(MappingTable& mapping_table) override;
  bool has_bias = true;
 protected:
  virtual void initialize_instructions(Tile* tile, Mapping mapping);
 private:
};