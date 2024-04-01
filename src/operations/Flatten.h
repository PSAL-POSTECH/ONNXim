/*TODO: implement this */
#pragma once

#include "Operation.h"

class Flatten : public Operation {
	public:
		Flatten(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
		Flatten(const Flatten& src);
		virtual void initialize_tiles(MappingTable& mapping_table) override;
    virtual void initialize_instructions(Tile* tile, Mapping mapping) override;
	protected:

	private:
		uint32_t _axis;
};