#include "Softmax.h"
#include "../Model.h"

Softmax::Softmax(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {

    /* Load weight info from node */
    _input_shape = get_input(0)->get_dims();

    assert(_input_shape.size()==2);
    _seq = _input_shape.at(0);
    _dk = _input_shape.at(1);

    for (int i=0;i<node_proto.output().size();i++) {
        _output_shape = _input_shape;
        if (node_proto.output(i)=="")
            continue;

        Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(i));
        if (pre_defind_tensor == nullptr) {
            std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
                _id, node_proto.output(i), _output_shape, _config.precision, false);
                _outputs.push_back(output_tensor.get()->get_id());
            _model->add_tensor(std::move(output_tensor));
        } else {
            pre_defind_tensor->redefine_tensor(_id, _output_shape);
        }
    }
    calculate_loops();
}

Softmax::Softmax(SimulationConfig config, MappingTable& mapping_table,
           std::vector<uint32_t> input_shape, uint32_t target_core)
    : Operation(config, mapping_table, target_core) {
    _input_shape = input_shape;
    _output_shape = input_shape;

    assert(_input_shape.size()==2);
    _seq = _input_shape.at(0);
    _dk = _input_shape.at(1);
    calculate_loops();
}

void Softmax::initialize_tiles(MappingTable& mapping_table) {
    for (uint32_t tokens=0; tokens < _seq; tokens+=_tokens_per_tile) {
        uint32_t remain_tokens = std::min(_seq-tokens, _tokens_per_tile);
        std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
            .status = Tile::Status::INITIALIZED,
            .optype = get_name(),
            .layer_id = _id,
            .accum = false,
        });
        /* dummy mapping */
        Mapping mapping;
        _tiles.push_back(std::move(tile));
        initialize_instructions(_tiles.back().get(), mapping, tokens, remain_tokens);

    }
}

void Softmax::initialize_instructions(Tile* tile, Mapping mapping, uint32_t token_offset, uint32_t tokens) {
    addr_type sram_base = SPAD_BASE;

    /* Load two tile (input: tokens x _dk, skip: tokens x _dk) */
    std::set<addr_type> dram_addrs;
    int offset;
    for (offset=0; offset<tokens*_dk*_config.precision; offset+=_config.dram_req_size)
        dram_addrs.insert(token_offset*_dk*_config.precision + offset);

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_base,
        .size = (uint32_t)dram_addrs.size(),
        .src_addrs = std::vector<addr_type>(dram_addrs.begin(), dram_addrs.end()),
        .operand_id = _INPUT_OPERAND,  // query
    }));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::SOFTMAX,
        .dest_addr = sram_base,
        .size = _dk * _config.precision,
        .src_addrs = std::vector<addr_type>{sram_base},
        .tile_m = tokens,
    }));

    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVOUT,
        .dest_addr = sram_base,
        .size = (uint32_t)dram_addrs.size(),
        .src_addrs = std::vector<addr_type>(dram_addrs.begin(), dram_addrs.end()),
        .operand_id = _OUTPUT_OPERAND,
    }));
}

void Softmax::calculate_loops() {
    uint32_t size_per_token = _dk * 2 * _config.precision;
    uint32_t sram_capacity = _config.core_config[target_core].spad_size KB / 2;  // unit: byte

    _tokens_per_tile = sram_capacity / size_per_token;
    assert (_tokens_per_tile >= 1);
    if (_tokens_per_tile > _seq) _tokens_per_tile = _seq;

    spdlog::info("[Softmax] tokens_per_tile: {}", _tokens_per_tile);
}