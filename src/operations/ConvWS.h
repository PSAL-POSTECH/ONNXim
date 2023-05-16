#pragma once

#include "Conv.h"

class ConvWS : public Conv {
 public:
  ConvWS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
  ConvWS(const Conv& src);
  ConvWS(SimulationConfig config, MappingTable mapping_table, convInfo info);
  virtual void initialize_tiles(MappingTable mapping_table) override;

 protected:
  virtual void initialize_instructions(Tile& tile, Mapping mapping);
  virtual void initialize_matmul_instructions(Tile& tile);

  void init(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
  Instruction make_weight_instruction(int m_offset, int s_offset, int r_offset,
                                      int c_offset, Mapping mapping);
  Instruction make_input_instruction(int m_offset, int s_offset, int r_offset,
                                     int c_offset, Mapping mapping);
};