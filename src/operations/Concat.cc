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
				_id, node_proto.output(0), output_shape, _config.precision, false);
		_outputs.push_back(output_tensor.get()->get_id());
		_model->add_tensor(std::move(output_tensor));
	} else {
		predefined_tensor->redefine_tensor(_id, output_shape);
	}
}

Concat::Concat(const Concat& src) : Operation(src) {
	_axis = src._axis;
}

void Concat::initialize_tiles(MappingTable& mapping_table) {
	spdlog::trace("initialize_tile {} ", _name);
	std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
		.status = Tile::Status::INITIALIZED,
		.optype = "Concat",
		.layer_id = _id
	});
	_tiles.push_back(std::move(tile));
}

void Concat::initialize_instructions(Tile* tile, Mapping mapping) {
}

