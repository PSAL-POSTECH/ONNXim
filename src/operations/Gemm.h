#pragma once
#include "Operation.h"

class Gemm : public Operation {
 public:
  Gemm(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
  Gemm(SimulationConfig config, MappingTable mapping_table,
       std::vector<uint32_t> output_shape, std::vector<uint32_t> input_shape,
       std::vector<uint32_t> weight_shape);

 protected:
  addr_type make_activation_address(uint32_t N, uint32_t H, uint32_t W,
                                             uint32_t C, std::vector<uint32_t> shape);

  std::vector<uint32_t> _output_shape;
  std::vector<uint32_t> _input_shape;
  std::vector<uint32_t> _weight_shape;

 private:
  uint32_t _alpha;
  uint32_t _beta;
  bool _transA;
  bool _transB;
};