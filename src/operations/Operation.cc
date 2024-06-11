#include "Operation.h"

#include "../Model.h"

Operation::Operation(SimulationConfig config, Model* model,
                     onnx::NodeProto& node_proto) {
  _id = generate_id();
  _model = model;
  _optype = node_proto.op_type();
  _name = node_proto.name();
  _proto = node_proto;
  _finish = false;
  _config = config;
  spdlog::trace("Node {} op_type {}", _name.c_str(), _optype.c_str());
  for (std::string input_proto : node_proto.input()) {
    /* Skip none input */
    if (input_proto == "")
      continue;

    Tensor* input_tensor = _model->find_tensor(input_proto);
    if (input_tensor == nullptr) {
      std::vector<uint32_t> dims;
      auto new_tensor = std::make_unique<Tensor>(0, input_proto, dims, _config.precision, false);
      input_tensor = new_tensor.get();
      _model->add_tensor(std::move(new_tensor));
    }
    _inputs.push_back(input_tensor->get_id());
    input_tensor->add_child_node(this);
  }
  if (_config.layout == "NCHW") {
    Ndim = 0;
    Cdim = 1;
    Hdim = 2;
    Wdim = 3;
  } else if (_config.layout == "NHWC") {
    Ndim = 0;
    Cdim = 3;
    Hdim = 1;
    Wdim = 2;
  }
  Mdim = 0;
  Cdim_w = 1;
  Sdim = 2;
  Rdim = 3;
}

Operation::Operation(SimulationConfig config, MappingTable& mapping_table) {
  _id = generate_id();
  _finish = false;
  _config = config;
  spdlog::trace("Node {} op_type {}", _name.c_str(), _optype.c_str());
  if (_config.layout == "NCHW") {
    Ndim = 0;
    Cdim = 1;
    Hdim = 2;
    Wdim = 3;
  } else if (_config.layout == "NHWC") {
    Ndim = 0;
    Cdim = 3;
    Hdim = 1;
    Wdim = 2;
  }
  Mdim = 0;
  Cdim_w = 1;
  Sdim = 2;
  Rdim = 3;
}

Operation::Operation(SimulationConfig config, Model* model,
                     onnx::NodeProto& node_proto, uint32_t id) {
  _id = id;
  _model = model;
  _optype = node_proto.op_type();
  _name = node_proto.name();
  _proto = node_proto;
  _finish = false;
  _config = config;
  spdlog::trace("Node {} op_type {}", _name.c_str(), _optype.c_str());
  for (std::string input_proto : node_proto.input()) {
    Tensor* input_tensor = _model->find_tensor(input_proto);
    if (input_tensor == nullptr) {
      std::vector<uint32_t> dims;
      auto new_tensor = std::make_unique<Tensor>(0, input_proto, dims, _config.precision, false);
      input_tensor = new_tensor.get();
      _model->add_tensor(std::move(new_tensor));
    }
    _inputs.push_back(input_tensor->get_id());
    input_tensor->add_child_node(this);
  }
}

Operation::Operation(SimulationConfig config, Model* model,
            std::string name,  std::map<std::string, std::string>&attribute) 
    : _config(config), _model(model) ,_name(name), _attributes(attribute) {
  _id = generate_id();
  _finish = false;
  Ndim = 0;
  Cdim = 3;
  Hdim = 1;
  Wdim = 2;
  Mdim = 0;
  Cdim_w = 1;
  Sdim = 2;
  Rdim = 3;
}

Operation::Operation(const Operation& operation) {
  spdlog::error("Opertion copy is not allowed !");
  exit(EXIT_FAILURE);
}

void Operation::set_finish() {
  for (auto id : _outputs) {
    Tensor* output = _model->get_tensor(id);
    output->set_produced();
  }
  _finish = true;
  spdlog::trace("layer {} finish", _name.c_str());
}

std::vector<uint32_t> Operation::get_child_nodes() {
  std::vector<uint32_t> result;
  for (auto id : _outputs) {
    Tensor* output = _model->get_tensor(id);
    spdlog::trace("num child nodes {}", output->num_child_nodes());
    for (int child = 0; child < output->num_child_nodes(); child++) {
      result.push_back(output->get_child_node(child));
    }
  }
  return result;
}

Tensor* Operation::get_input(int id) { return _model->get_tensor(_inputs.at(id)); }

Tensor* Operation::get_output(int id) {
  return _model->get_tensor(_outputs.at(id));
}

void Operation::add_input(int id) { 
  _inputs.push_back(id);
  _model->get_tensor(id)->add_child_node(this);
}

void Operation::add_output(int id) { 
  _outputs.push_back(id);
}

bool Operation::check_executable() {
  bool result = true;
  for (auto id : _inputs) {
    Tensor* input = _model->get_tensor(id);
    result = result && input->get_produced();
    spdlog::trace("Layer {}: Input {} Produced {}", _name.c_str(),
                  input->get_name().c_str(), input->get_produced());
  }
  return result;
}

std::deque<std::unique_ptr<Tile>>& Operation::get_tiles() { //TODO: fix the return _tiles to new_tile
  return _tiles;
}

void Operation::clear_tiles() { //TODO: fix the return _tiles to new_tile
  _tiles.clear();
}

void Operation::initialize_tiles(MappingTable& mapping_table) {
  spdlog::trace("Parent");
}

addr_type Operation::get_operand_addr(uint32_t operand_id) {
  if (operand_id == _NO_OPERAND)
    return GARBEGE_ADDR;
  else if (operand_id >= _INPUT_OPERAND && operand_id < _OUTPUT_OPERAND) {
    if ((operand_id - _INPUT_OPERAND) >= _inputs.size())
      return (addr_type) 0;
    return get_input(operand_id - _INPUT_OPERAND)->get_address();
  } else if (operand_id >= _OUTPUT_OPERAND) {
    if ((operand_id - _OUTPUT_OPERAND) >= _outputs.size())
      return (addr_type) 0;
    return get_output(operand_id - _OUTPUT_OPERAND)->get_address();
  } else {
    assert(0);
    return GARBEGE_ADDR;
  }
}

addr_type Operation::make_activation_address(uint32_t N, uint32_t H, uint32_t W,
                                             uint32_t C,
                                             std::vector<uint32_t> shape) {
  addr_type address;
  if (_config.layout == "NCHW") {
    address = (N * shape[Cdim] * shape[Hdim] * shape[Wdim] +
               C * shape[Hdim] * shape[Wdim] + H * shape[Wdim] + W) *
              _config.precision;
  } else if (_config.layout == "NHWC") {
    address = (N * shape[Hdim] * shape[Wdim] * shape[Cdim] +
               H * shape[Wdim] * shape[Cdim] + W * shape[Cdim] + C) *
              _config.precision;
  }
  return _config.align_address(address);
}

addr_type Operation::make_weight_address(uint32_t S, uint32_t R, uint32_t M,
                                         uint32_t C,
                                         std::vector<uint32_t> shape) {
  addr_type address;
  int padded_C =
      shape[Cdim_w] + (_config.core_width - shape[Cdim_w] % _config.core_width);
  // int padded_S = shape[Cdim] + (_config.core_width - shape[Cdim] %
  // _config.core_width);
  if (_config.layout == "NCHW") {
    address = (M * shape[Cdim_w] * shape[Sdim] * shape[Rdim] +
               C * shape[Sdim] * shape[Rdim] + S * shape[Rdim] + R) *
              _config.precision;
  } else if (_config.layout == "NHWC") {
    address = ((M / _config.core_width) * shape[Sdim] * shape[Rdim] * padded_C +
               S * shape[Rdim] * padded_C + R * padded_C + C) *
              _config.precision * _config.core_width;
  }
  return _config.align_address(address);
}

std::string Operation::get_attribute(std::string key) {
  if (_attributes.find(key) == _attributes.end()) {
    spdlog::error("{}: Attribute {} not found", _name, key.c_str());
    exit(EXIT_FAILURE);
  }
  return _attributes[key];
}