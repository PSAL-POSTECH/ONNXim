#pragma once

#include "Conv.h"

class ConvWS : public Conv {
 public:
  ConvWS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
  ConvWS(const Conv& src);
  ConvWS(SimulationConfig config, MappingTable& mapping_table, convInfo info);
  virtual void initialize_tiles(MappingTable& mapping_table) override;

 protected:
  virtual void initialize_instructions(Tile* tile, Mapping mapping);
  virtual void initialize_matmul_instructions(Tile* tile);
  virtual addr_type make_weight_address(uint32_t S, uint32_t R, uint32_t M, uint32_t C,
                                        std::vector<uint32_t> shape);
  virtual addr_type make_activation_address(uint32_t N, uint32_t H, uint32_t W,
                                            uint32_t C, std::vector<uint32_t> shape);
  void init(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
  Instruction make_weight_instruction(int m_offset, int s_offset, int r_offset,
                                      int c_offset, Mapping mapping);
  Instruction make_input_instruction(int m_offset, int s_offset, int r_offset,
                                     int c_offset, Mapping mapping);
};