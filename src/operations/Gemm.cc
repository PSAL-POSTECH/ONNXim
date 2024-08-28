#include "Gemm.h"

#include "../Model.h"
#include "../Tensor.h"

Gemm::Gemm(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  Mdim = 1;
  Cdim_w = 0;
  Cdim = 1;
  Ndim = 0;

  for (auto attribute : node_proto.attribute()) {
    if (attribute.name() == "alpha") {
      _alpha = attribute.f();
    } else if (attribute.name() == "beta") {
      _beta = attribute.f();
    } else if (attribute.name() == "transA") {
      _transA = attribute.i();
      if (_transA) {
        Cdim = 0;
        Ndim = 1;
      }
    } else if (attribute.name() == "transB") {
      _transB = attribute.i();
      if (_transB) {
        Mdim = 0;
        Cdim_w = 1;
      }
    }
  }

  _input_shape = get_input(0)->get_dims();
  _weight_shape = get_input(1)->get_dims();
  _output_shape = _input_shape;
  _output_shape[_input_shape.size()-2+Ndim] = _input_shape[_input_shape.size()-2 + Ndim];
  _output_shape[_input_shape.size()-2+Cdim] = _weight_shape[Mdim];

  _batch_size = 1;
  for (int i=0; i<_input_shape.size()-2;i++)
    _batch_size *= _input_shape.at(i);

  spdlog::trace("GemmWS: input_shape: {}", _input_shape);
  spdlog::trace("GemmWS: output_shape : {}", _output_shape);

  std::vector<uint32_t> bias_shape;
  if (node_proto.input().size() == 3) {
    bias_shape = get_input(2)->get_dims();
    assert(bias_shape[0] == _output_shape[_input_shape.size()-2+Cdim]);
  }

  Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(0));
  if (pre_defind_tensor == nullptr) {
    std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
        _id, node_proto.output(0), _output_shape, _config.precision, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  } else {
    pre_defind_tensor->redefine_tensor(_id, _output_shape);
  }
}

Gemm::Gemm(SimulationConfig config, MappingTable& mapping_table,
           std::vector<uint32_t> input_shape,
           std::vector<uint32_t> weight_shape,
           std::vector<uint32_t> output_shape, uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
  Mdim = 1;
  Cdim_w = 0;
  Cdim = 1;
  Ndim = 0;
  _input_shape = input_shape;
  _weight_shape = weight_shape;
  _output_shape = output_shape;

  _batch_size = 1;
  for (int i=0; i<_input_shape.size()-2;i++)
    _batch_size *= _input_shape.at(i);

  spdlog::debug("[Gemm] input_shape: {}", _input_shape);
  spdlog::debug("[Gemm] output_shape : {}", _output_shape);
}

Gemm::Gemm(SimulationConfig config, Model* model, std::string name,
            std::map<std::string, std::string> &attributes, uint32_t target_core) 
    :Operation(config, model, name, attributes, target_core) {
  Mdim = 1;
  Cdim_w = 0;
  Cdim = 1;
  Ndim = 0;
  _batch_size = 1;

  _input_shape = parse_dims(get_attribute("input_shape"));
  _output_shape = parse_dims(get_attribute("output_shape"));
  _weight_shape = parse_dims(get_attribute("weight_shape"));
  std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
        _id, name_gen(_name, "out"), _output_shape, _config.precision, false);
  _outputs.push_back(output_tensor.get()->get_id());
  _model->add_tensor(std::move(output_tensor));
  spdlog::debug("[Gemm] input_shape: {}", _input_shape);
  spdlog::debug("[Gemm] output_shape : {}", _output_shape);
}