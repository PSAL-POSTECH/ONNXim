/*TODO: implement this */
#include "Concat.h"

#include "../Model.h"
#include "../Tensor.h"

Concat::Concat(SimulationConfig config, Model* model,
              	onnx::NodeProto& node_proto) 
    : Operation(config, model, node_proto) {
	for (auto attribute : node_proto.attribute()) {
		if (attribute.name() == "axis") {
			spdlog::trace("concat axis {}", attribute.ints(0));
      _axis = attribute.ints(0);
		} 
	}

	assert(_axis>=0 && _axis<4);
	std::vector<uint32_t> output_shape;
	std::vector<uint32_t> input0_shape = get_input(0)->get_dims();
	std::vector<uint32_t> input1_shape = get_input(1)->get_dims();
	output_shape.resize(input0_shape.size());
	for (int i = 0; i < input0_shape.size(); i++) {
		if (i == _axis)
			continue;
		assert(input0_shape[i] == input1_shape[i]);
		output_shape[i] = input0_shape[i];
	}
	output_shape[_axis] = input0_shape[_axis] + input1_shape[_axis];

	spdlog::trace("output name : {} {}", node_proto.output(0).c_str(), 
									output_shape);
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

Concat::Concat(const Concat& src) : Operation(src) {
	_axis = src._axis;
}

void Concat::initialize_tiles(MappingTable mapping_table) {
	spdlog::trace("initialize_tile {} ", _name);
		
	_tiles.push_back(Tile{.status = Tile::Status::INITIALIZED,
												.optype = "Concat",
												.layer_id = _id});
}

void Concat::initialize_instructions(Tile& tile, Mapping mapping) {
// 	std::vector<uint32_t> output_shape = get_output(0)->get_dims();
// 	std::vector<uint32_t> input0_shape = get_input(0)->get_dims();
// 	std::vector<uint32_t> input1_shape = get_input(1)->get_dims();

// 	std::set<addr_type> input_set;
// 	std::string input_id = fmt::format("INPUT-{}", tile.layer_id);

// 	for (int N = 0; N < input0_shape[Ndim]; N++) {
// 		for (int C = 0; C < input0_shape[Cdim]; C++) {
// 			for (int Q = 0; Q < input0_shape[Hdim]; Q++) {
// 				for (int P = 0; P < input0_shape[Wdim]; P++) {
// 					input_set.insert(make_activation_address(N, Q, P, C, input0_shape));
// 				}
// 			}
// 		}
// 	}

// 	for (int N = 0; N < input1_shape[Ndim]; N++) {
// 		for (int C = 0; C < input1_shape[Cdim]; C++) {
// 			for (int Q = 0; Q < input1_shape[Hdim]; Q++) {
// 				for (int P = 0; P < input1_shape[Wdim]; P++) {
// 					input_set.insert(make_activation_address(N, Q, C, P, input1_shape));
// 				}
// 			}
// 		}
// 	}

// 	tile.instructions.push_back(
// 				Instruction{.opcode = Opcode::MOVIN,
// 										.id = input_id,
// 										.addrs = std::vector<addr_type>(
// 													input_set.begin(), input_set.end())});

// 	std::set<addr_type> output_set;
// 	std::string output_id = fmt::format("OUT-{}", tile.layer_id);	

// 	output_set.insert(make_activation_address(0, 0, 0, 0, output_shape));

// 	tile.instructions.push_back(
// 				Instruction{.opcode = Opcode::MOVOUT,
// 										.id = output_id,
// 										.dependent_ids = std::vector<std::string>{input_id},
// 										.addrs = std::vector<addr_type>(
// 													output_set.begin(), output_set.end())});
}

