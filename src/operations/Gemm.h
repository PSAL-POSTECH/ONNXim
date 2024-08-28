#pragma once
#include "Operation.h"

class Gemm : public Operation {
 public:
  Gemm(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
  Gemm(SimulationConfig config, MappingTable& mapping_table,
       std::vector<uint32_t> output_shape, std::vector<uint32_t> input_shape,
       std::vector<uint32_t> weight_shape, uint32_t target_core=0);
  Gemm(SimulationConfig config, Model* model, std::string name,
        std::map<std::string, std::string>& attributes, uint32_t target_core=0);

 protected:

  std::vector<uint32_t> _output_shape;
  std::vector<uint32_t> _input_shape;
  std::vector<uint32_t> _weight_shape;
  int _batch_size;

 private:
  uint32_t _alpha;
  uint32_t _beta;
  bool _transA;
  bool _transB;
};