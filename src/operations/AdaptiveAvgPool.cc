#include "AdaptiveAvgPool.h"

#include "../Model.h"
#include "../Tensor.h"

AdaptiveAvgPool::AdaptiveAvgPool(SimulationConfig config, Model* model,
                                 onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  int kernel_dim = 0;
  for (auto attribute : node_proto.attribute()) {
    if (attribute.name() == "kernel_shape") {
      spdlog::trace(" kernel_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _kernel_shape.push_back(attribute.ints(i));
      }
      kernel_dim = attribute.ints_size();
    } else if (attribute.name() == "strides") {
      spdlog::trace("stride_shape {}", attribute.ints_size());
      for (int i = 0; i < attribute.ints_size(); i++) {
        _strides.push_back(attribute.ints(i));
      }
    }
  }

  /* We assume AdaptiveAvgPool2d */
  assert(kernel_dim == 2);
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();
  std::vector<uint32_t> output_shape = input_shape;

  /* Asuming input H W size are multiple of output H W*/
  assert(!(input_shape[Hdim] % _kernel_shape[0]) &&
         !(input_shape[Wdim] % _kernel_shape[1]));

  output_shape[Hdim] = (input_shape[Hdim] - _kernel_shape[0]) / _strides[0] + 1;
  output_shape[Wdim] = (input_shape[Wdim] - _kernel_shape[1]) / _strides[1] + 1;

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
}

AdaptiveAvgPool::AdaptiveAvgPool(const AdaptiveAvgPool& src) : Operation(src) {
  _kernel_shape = src._kernel_shape;
  _strides = src._strides;
  _skip = src._skip;
}

void AdaptiveAvgPool::initialize_tiles(MappingTable& mapping_table) {
  spdlog::trace("initialize_tile {}", _name);
  std::vector<uint32_t> output_shape = get_output(0)->get_dims();
  if (_skip) {
    _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED, .skip = true}));
    return;
  }

  std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
    .status = Tile::Status::INITIALIZED,
    .optype = "AdaptiveAvgPool",
    .layer_id = _id,
    .skip = true});
  _tiles.push_back(std::move(tile));
  initialize_instructions(_tiles.back().get(), Mapping{});
}

void AdaptiveAvgPool::initialize_instructions(Tile* tile, Mapping mapping) {
  return;
}