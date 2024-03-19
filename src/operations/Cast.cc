#include "Cast.h"
#include "../Model.h"
#include "../Tensor.h"

Cast::Cast(SimulationConfig config, Model* model, onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {
  _input_shape = get_input(0)->get_dims();
  _output_shape = _input_shape;

  Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(0));
  if (pre_defind_tensor == nullptr) {
    std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
        _id, node_proto.output(0), _output_shape, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  } else {
    pre_defind_tensor->redefine_tensor(_id, _output_shape);
  }
  /* Calculate tile size */
  _total_loop = std::accumulate(_input_shape.begin(), _input_shape.end(), 1U, [](uint32_t a, uint32_t b) {
    return (uint64_t)a * (uint64_t)b;
  });
  _element_in_tile = _config.sram_size / (config.precision * 2); // Doubled buffer
}

void Cast::initialize_tiles(MappingTable mapping_table) {
  for (uint64_t i=0; i<_total_loop; i+=_element_in_tile) {
    uint32_t remainder = std::min(_element_in_tile, (uint32_t(_total_loop - i)));
    _tiles.push_back(Tile{.status = Tile::Status::INITIALIZED,
                          .optype="Cast",
                          .layer_id=_id,
                          .C = remainder,
                          .skip = true});
    initialize_instructions(_tiles.back(), Mapping{});
  }
}

void Cast::initialize_instructions(Tile& tile, Mapping mapping) {
}