/*TODO: implement this */
#include "Flatten.h"

#include "../Model.h"
#include "../Tensor.h"

Flatten::Flatten(SimulationConfig config, Model* model,
                 onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  for (auto attribute : node_proto.attribute()) {
    if (attribute.name() == "axis") {
      spdlog::trace("flatten axis {}", attribute.i());
      _axis = attribute.i();
    }
  }

  assert(_axis >= 0 && _axis < 4);
  std::vector<uint32_t> input_shape = get_input(0)->get_dims();
  std::vector<uint32_t> output_shape(_axis + 1, 1);

  for (int i = 0; i < input_shape.size(); i++) {
    if (i < _axis) {
      output_shape[i] = input_shape[i];
    } else {
      output_shape[_axis] *= input_shape[i];
    }
  }

  spdlog::trace("output name : {} {}", node_proto.output(0).c_str(), output_shape);

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

Flatten::Flatten(const Flatten& src) : Operation(src) { _axis = src._axis; }

void Flatten::initialize_tiles(MappingTable& mapping_table) {
  spdlog::trace("initialize_tile {}", _name);

  _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED,
                        .optype = "Flatten",
                        .layer_id = _id,
                        .skip = true}));
  initialize_instructions(_tiles.back().get(), Mapping{});
}

void Flatten::initialize_instructions(Tile* tile, Mapping mapping) {
}
