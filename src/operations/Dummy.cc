#include "Dummy.h"
#include "../Model.h"
#include "../Tensor.h"

Dummy::Dummy(SimulationConfig config, Model* model, onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {
  _input_shape = get_input(0)->get_dims();
  _output_shape = _input_shape;
  spdlog::trace("output_shape : {}", _output_shape);
  spdlog::trace("output name : {} {}", node_proto.output(0).c_str());

  for (int i=0;i<node_proto.output().size();i++) {
    Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(i));
    if (pre_defind_tensor == nullptr) {
      std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
          _id, node_proto.output(i), _output_shape, false);
      _outputs.push_back(output_tensor.get()->get_id());
      _model->add_tensor(std::move(output_tensor));
    } else {
      pre_defind_tensor->redefine_tensor(_id, _output_shape);
    }
  }
}

void Dummy::initialize_tiles(MappingTable mapping_table) {
  _tiles.push_back(Tile{.status = Tile::Status::INITIALIZED,
                        .optype="Dummy",
                        .layer_id=_id,
                        .skip = true});
  initialize_instructions(_tiles.back(), Mapping{});
}

void Dummy::initialize_instructions(Tile& tile, Mapping mapping) {
}