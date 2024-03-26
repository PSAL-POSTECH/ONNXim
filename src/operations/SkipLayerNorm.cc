#include "SkipLayerNorm.h"
#include "../Model.h"

SkipLayerNorm::SkipLayerNorm(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {

    /* Load weight info from node */
    _input_shape = get_input(0)->get_dims();
    _skip_shape = get_input(1)->get_dims();
    _weight_shape = get_input(2)->get_dims();
    _bias_shape = get_input(3)->get_dims();
    _dense_bias_shape = get_input(3)->get_dims();

    assert(_input_shape.size()==3);
    _batch_size = _input_shape.at(0);
    _seq = _input_shape.at(1);
    _dk = _input_shape.at(2);

    for (int i=0;i<node_proto.output().size();i++) {
        _output_shape = _input_shape;
        if (node_proto.output(i)=="")
            continue;

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
    calculate_loops();
}

void SkipLayerNorm::initialize_tiles(MappingTable mapping_table) {
    for (uint32_t tokens=0; tokens < _seq*_batch_size; tokens+=_tokens_per_tile) {
        uint32_t remain_tokens = std::min(_seq*_batch_size-tokens, _tokens_per_tile);
        auto tile = Tile{
            .status = Tile::Status::INITIALIZED,
            .optype = get_name(),
            .layer_id = _id,
            .accum = false,
        };
        /* dummy mapping */
        Mapping mapping;
        initialize_instructions(tile, mapping, tokens, remain_tokens);

        _tiles.push_back(tile);
    }
}

void SkipLayerNorm::initialize_instructions(Tile &tile, Mapping mapping, uint32_t token_offset, uint32_t tokens) {
    addr_type sram_base = SPAD_BASE;
    addr_type sram_bias_base = sram_base + _batch_size * _seq * _dk * _config.precision;

    /* Load two tile (input: tokens x _dk, skip: tokens x _dk) */
    std::set<addr_type> dram_addrs;
    std::set<addr_type> dram_skip_addrs;
    int offset;
    for (offset=0; offset<tokens*_dk*_config.precision; offset+=_config.dram_req_size)
        dram_addrs.insert(token_offset*_dk*_config.precision + offset);
    for (;offset<tokens*_dk*_config.precision*2; offset+=_config.dram_req_size)
        dram_skip_addrs.insert(token_offset*_dk*_config.precision + offset);

    tile.instructions.push_back(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_base,
        .size = (uint32_t)dram_addrs.size(),
        .src_addrs = std::vector<addr_type>(dram_addrs.begin(), dram_addrs.end()),
        .operand_id = _INPUT_OPERAND,  // query
    });

    tile.instructions.push_back(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_bias_base,
        .size = (uint32_t)dram_skip_addrs.size(),
        .src_addrs = std::vector<addr_type>(dram_skip_addrs.begin(), dram_skip_addrs.end()),
        .operand_id = _INPUT_OPERAND+1,  // query
    });

    tile.instructions.push_back(Instruction{
        .opcode = Opcode::LAYERNORM,
        .dest_addr = sram_base,
        .size = _dk * _config.precision,
        .src_addrs = std::vector<addr_type>{sram_base},
        .tile_m = tokens,
    });

    tile.instructions.push_back(Instruction{
        .opcode = Opcode::ADD,
        .dest_addr = sram_base,
        .size = tokens * _dk * _config.precision,
        .src_addrs = std::vector<addr_type>{sram_base, sram_bias_base},
    });

    tile.instructions.push_back(Instruction{
        .opcode = Opcode::MOVOUT,
        .dest_addr = sram_base,
        .size = (uint32_t)dram_addrs.size(),
        .src_addrs = std::vector<addr_type>{sram_base},
        .operand_id = _OUTPUT_OPERAND,
    });
}

void SkipLayerNorm::calculate_loops() {
    uint32_t size_per_token = _dk * 2 * _config.precision;
    uint32_t sram_capacity = _config.spad_size KB / 2;  // unit: byte

    _tokens_per_tile = sram_capacity / size_per_token;
    assert (_tokens_per_tile >= 1);
    if (_tokens_per_tile > _seq * _batch_size) _tokens_per_tile = _seq * _batch_size;

    spdlog::info("[SkipLayerNorm] tokens_per_tile: {}", _tokens_per_tile);
}