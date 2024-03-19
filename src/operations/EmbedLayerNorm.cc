#include "EmbedLayerNorm.h"
#include "../Model.h"
#include "../Tensor.h"

EmbedLayerNorm::EmbedLayerNorm(SimulationConfig config, Model* model, onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {
  _input_shape = get_input(0)->get_dims();
  _weight_shape = get_input(2)->get_dims();

  _output_shape.push_back(_input_shape.at(1)); 
  _output_shape.push_back(_weight_shape.at(1)); 
  spdlog::trace("output_shape : {}", _output_shape);

  /* output */
  Tensor* embed_output = _model->find_tensor(node_proto.output(0));
  if (embed_output == nullptr) {
    std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
        _id, node_proto.output(0), _output_shape, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  } else {
    embed_output->redefine_tensor(_id, _output_shape);
  }

  /* mask */
  Tensor* mask_output = _model->find_tensor(node_proto.output(1));
  if (mask_output == nullptr) {
    std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
        _id, node_proto.output(1), _output_shape, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  } else {
    mask_output->redefine_tensor(_id, _output_shape);
  }
}

void EmbedLayerNorm::initialize_tiles(MappingTable mapping_table) {
  _tiles.push_back(Tile{.status = Tile::Status::INITIALIZED,
                        .optype="EmbedLayerNorm",
                        .layer_id=_id,
                        .skip=true});
}

void EmbedLayerNorm::initialize_instructions(Tile& tile, Mapping mapping) {
}