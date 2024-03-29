#pragma once

#include "Operation.h"

struct convInfo{
  std::vector<uint32_t> kernel_shape;
  std::vector<uint32_t> strides;
  std::vector<uint32_t> dilations;
  std::vector<uint32_t> pads;
  std::vector<uint32_t> input_shape;
  std::vector<uint32_t> weight_shape;
  std::vector<uint32_t> conv_out_shape;
  std::vector<uint32_t> pool_out_shape;
  uint32_t group;
  bool activation_fused;
  std::string activation_type;
  bool bathnorm_fused;
  bool skip_connection_fused;
  bool pool_fused;
  std::string pool_type;
  std::vector<uint32_t> pool_kernel_shape;
  std::vector<uint32_t> pool_strides;
  std::vector<uint32_t> pool_pads;  
};

class Conv : public Operation {
  public:
    Conv(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
    Conv(const Conv& src);
    Conv(SimulationConfig config, MappingTable& mapping_table, convInfo info);
    // virtual void initialize_tiles(MappingTable& mapping_table) override;
  protected:
    virtual void im2col_nhwc();
    // void init(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);

  protected:
    std::vector<uint32_t> _kernel_shape;
    std::vector<uint32_t> _strides;
    std::vector<uint32_t> _dilations;
    std::vector<uint32_t> _pads;  
    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _weight_shape;
    std::vector<uint32_t> _conv_out_shape;
    std::vector<uint32_t> _pool_out_shape;
    uint32_t _group;
    bool _activation_fused;
    std::string _activation_type;
    bool _bathnorm_fused;
    bool _skip_connection_fused;
    bool _pool_fused;
    std::string _pool_type;
    std::vector<uint32_t> _pool_kernel_shape;
    std::vector<uint32_t> _pool_strides;
    std::vector<uint32_t> _pool_pads;  

};