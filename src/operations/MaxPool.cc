#include "MaxPool.h"

#include <robin_hood.h>

#include "../Model.h"
#include "../Tensor.h"

MaxPool::MaxPool(SimulationConfig config, Model* model,
                 onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  int kernel_dim = 0;
  for (auto attribute : node_proto.attribute()) {
    if (attribute.name() == "kernel_shape") {
      spdlog::trace("kernel_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _kernel_shape.push_back(attribute.ints(i));
      }
      kernel_dim = attribute.ints_size();
    } else if (attribute.name() == "strides") {
      for (int i = 0; i < attribute.ints_size(); i++) {
        _strides.push_back(attribute.ints(i));
      }
    } else if (attribute.name() == "auto_pad") {
    } else if (attribute.name() == "pads") {
      spdlog::trace("padn_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _pads.push_back(attribute.ints(i));
      }
    }
  }

  /* We assume conv2d */
  assert(kernel_dim == 2);
  std::vector<uint32_t> output_shape;
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();
  output_shape.resize(input_shape.size());
  output_shape[Ndim] = input_shape[Ndim];
  output_shape[Cdim] = input_shape[Cdim];
  for (int i = 0; i < kernel_dim; i++) {
    output_shape[Hdim + i] =
        (uint32_t)ceil(((float)input_shape[Hdim + i] + _pads[i] +
                        _pads[i + 2] - (_kernel_shape[i] - 1)) /
                       (float)_strides[i]);
  }

  spdlog::trace("output name : {} {}", node_proto.output(0).c_str(),
                output_shape);
  Tensor* predefined_tensor = _model->find_tensor(node_proto.output(0));
  if (predefined_tensor == nullptr) {
    std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
        _id, node_proto.output(0), output_shape, _config.precision, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  } else {
    predefined_tensor->redefine_tensor(_id, output_shape);
  }

  _tiles.push_back(
      std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED, .layer_id = _id, .batch = 0, .skip = true}));
}

MaxPool::MaxPool(const MaxPool& src) : Operation(src) {
  _kernel_shape = src._kernel_shape;
  _strides = src._strides;
  _dilations = src._dilations;
  _pads = src._pads;
}

/*TODO: implement this */
void MaxPool::initialize_tiles(MappingTable& mapping_table) {
  spdlog::trace("initialize_tile {} ", _name);
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();  

  uint32_t h_shift = (input_shape[Hdim] - _kernel_shape[0]) / _strides[0] + 1;
  uint32_t w_shift = (input_shape[Wdim] - _kernel_shape[1]) / _strides[1] + 1;

  _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED,
                        .optype = "MaxPool",
                        .layer_id = _id,
                        .skip = true}));
  initialize_instructions(_tiles.back().get(), Mapping{});
}

void MaxPool::initialize_instructions(Tile* tile, Mapping mapping) {
}