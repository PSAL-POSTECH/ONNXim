/*TODO: implement this */
#include "Flatten.h"

#include "../Model.h"
#include "../Tensor.h"

Flatten::Flatten(SimulationConfig config, Model* model,
                 onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {
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
        _id, node_proto.output(0), output_shape, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
  } else {
    predefined_tensor->redefine_tensor(_id, output_shape);
  }
}

Flatten::Flatten(const Flatten& src) : Operation(src) { _axis = src._axis; }

void Flatten::initialize_tiles(MappingTable& mapping_table) {
  spdlog::trace("initialize_tile {}", _name);

  _tiles.push_back(Tile{.status = Tile::Status::INITIALIZED,
                        .optype = "Flatten",
                        .layer_id = _id,
                        .skip = true});
  initialize_instructions(_tiles.back(), Mapping{});
}

void Flatten::initialize_instructions(Tile& tile, Mapping mapping) {
  // 	std::vector<uint32_t> output_shape = get_output(0)->get_dims();
  // 	std::vector<uint32_t> input_shape = get_input(0)->get_dims();

  // 	std::set<addr_type> input_set;
  // 	std::string input_id = fmt::format("INPUT-{}", tile.layer_id);

  // 	for (int N = 0; N < input_shape[Ndim]; N++) {
  // 		for (int C = 0; C < input_shape[Cdim]; C++) {
  // 			for (int Q = 0; Q < input_shape[Hdim]; Q++) {
  // 				for (int P = 0; P < input_shape[Wdim]; P++){
  // 					input_set.insert(make_activation_address(N, Q, P,
  // C, input_shape));
  // 				}
  // 			}
  // 		}
  // 	}

  // 	tile.instructions.push_back(
  // 							Instruction{.opcode =
  // Opcode::MOVIN, 													.id = input_id,
  // .addrs = std::vector<addr_type>( 																input_set.begin(), input_set.end())});

  // 	std::set<addr_type> output_set;
  // 	std::string output_id = fmt::format("OUT-{}", tile.layer_id);

  // 	output_set.insert(make_activation_address(0, 0, 0, 0, output_shape));

  // 	tile.instructions.push_back(
  // 				Instruction{.opcode = Opcode::MOVOUT,
  // 										.id
  // =
  // output_id, 										.dependent_ids =
  // std::vector<std::string>{input_id}, 										.addrs
  // =
  // std::vector<addr_type>(
  // output_set.begin(), output_set.end())});
}
