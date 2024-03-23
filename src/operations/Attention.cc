#include "Attention.h"
#include "../Model.h"
#include "../Tensor.h"
#include "GemmWS.h"

Attention::Attention(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {

    for (auto attribute : node_proto.attribute()) {
        if (attribute.name() == "num_heads") {
            _nh = attribute.i();
        }
    }

    /* Load weight info from node */
    _input_shape = get_input(0)->get_dims();
    _weight_shape = get_input(1)->get_dims();
    _bias_shape = get_input(2)->get_dims();
    _mask_shape = get_input(3)->get_dims();
    if (node_proto.input().size()==5) {
        _kv_cache_shape = get_input(4)->get_dims();
        /* If "past_seq_len" is not 0 */
        if (_kv_cache_shape.at(3))
            has_kv_cache = true;
    }
    assert(_input_shape.size()==3);
    _batch_size = _input_shape.at(0);
    _dmodel = _weight_shape.at(0);
    _dk = _dmodel / _nh;
    _q_len = _input_shape.at(1);
    if (has_kv_cache)
        _seq = _kv_cache_shape.at(3) + 1;
    else
        _seq = _input_shape.at(1);

    _query_shape = std::vector<uint32_t>{_nh, _q_len, _dk};
    _key_shape = std::vector<uint32_t>{_nh, _dk, _seq};
    _value_shape = std::vector<uint32_t>{_nh, _seq, _dk};

    _output_shape = std::vector<uint32_t>{_batch_size, _q_len, _dmodel};
    _liner_output_shape = std::vector<uint32_t>{_batch_size, _q_len, _weight_shape[1]};
    spdlog::debug("Fused attention: input shape: [{}, {}, {}]", _input_shape.at(0), _input_shape.at(1), _input_shape.at(2));
    spdlog::debug("Fused attention: output shape: [{}, {}, {}]", _output_shape.at(0), _output_shape.at(1), _output_shape.at(2));
    spdlog::debug("Fused attention: query shape: [{}, {}, {}]", _query_shape.at(0), _query_shape.at(1), _query_shape.at(2));
    spdlog::debug("Fused attention: key shape: [{}, {}, {}]", _key_shape.at(0), _key_shape.at(1), _key_shape.at(2));
    spdlog::debug("Fused attention: value shape: [{}, {}, {}]", _value_shape.at(0), _value_shape.at(1), _value_shape.at(2));

    Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(0));
    if (pre_defind_tensor == nullptr) {
        std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
            _id, node_proto.output(0), _output_shape, false);
            _outputs.push_back(output_tensor.get()->get_id());
        _model->add_tensor(std::move(output_tensor));
    } else {
        pre_defind_tensor->redefine_tensor(_id, _output_shape);
    }
    calculate_loops();
}

void Attention::initialize_tiles(MappingTable mapping_table) {
    /* linear projection */
    GemmWS linear_projection = GemmWS(_config, mapping_table, _input_shape, _weight_shape, _liner_output_shape);
    linear_projection.initialize_tiles(mapping_table);
    std::deque<Tile> tiles = linear_projection.get_tiles();
    for (Tile& tile : tiles) {
        tile.layer_id = _id;
        _tiles.push_back(tile);
    }

    /* Fused Attention body */
    for (int req_idx = 0; req_idx < _batch_size; req_idx++) {
        int heads_per_tile = _heads_per_tile[req_idx];
        for (int head_off=0; head_off<_nh; head_off+=heads_per_tile) {
            uint32_t remain_heads = std::min(_nh-head_off, (uint32_t)heads_per_tile);
            auto tile = Tile{
                .status = Tile::Status::INITIALIZED,
                .optype = get_name(),
                .layer_id = _id,
                //.K = 0,
                .accum = false,
            };
            /* dummy mapping */
            Mapping mapping;
            initialize_instructions(tile, mapping, head_off, heads_per_tile);

            _tiles.push_back(tile);
        }
    }
}

// 일단 한 tile에는 최대 하나의 request만 있는 경우부터.
void Attention::initialize_instructions(Tile &tile, Mapping mapping, int head_idx, int num_heads) {
    // head_idx # start idx
    // num_heads
    uint32_t q_len = _q_len;
    uint32_t seq_len = _seq;

    addr_type sram_query_base = SPAD_BASE;
    addr_type sram_key_base = sram_query_base + q_len * _dk * num_heads * _config.precision;
    addr_type sram_value_base = sram_key_base + _dk * seq_len * num_heads * _config.precision;
    addr_type sram_logit_base = ACCUM_SPAD_BASE;  // for logits

    for (int h_ofs = 0; h_ofs < num_heads; h_ofs++) {
        int h_idx = head_idx + h_ofs;

        addr_type sram_q_ofs = sram_query_base + h_ofs * (q_len * _dk) * _config.precision;
        addr_type sram_k_ofs = sram_key_base + h_ofs * (_dk * seq_len) * _config.precision;
        addr_type sram_v_ofs = sram_value_base + h_ofs * (_dk * seq_len) * _config.precision;
        addr_type sram_l_ofs = sram_logit_base + h_ofs * (q_len * seq_len) * _config.precision;

        std::set<addr_type> dram_query_addrs;  // = _query[req_idx]->get_all_addrs();
        std::set<addr_type> dram_key_addrs;    // = _key[req_idx]->get_all_addrs();
        std::set<addr_type> dram_value_addrs;
        std::set<addr_type> dram_output_addrs;

        for (int i = 0; i < _dk; i++) {
            for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
                // key:  h, d_k, seq_len
                std::vector<uint32_t> query_idx = {(uint32_t)(h_idx), (uint32_t)seq_idx, (uint32_t)i};
                std::vector<uint32_t> key_idx =   {(uint32_t)(h_idx+_nh), (uint32_t)i, (uint32_t)seq_idx};
                std::vector<uint32_t> value_idx = {(uint32_t)(h_idx+_nh*2), (uint32_t)seq_idx, (uint32_t)i};
                std::vector<uint32_t> output_idx = {(uint32_t)(h_idx+_nh*3), (uint32_t)seq_idx, (uint32_t)i};

                dram_key_addrs.insert(make_address(key_idx, _key_shape));
                dram_value_addrs.insert(make_address(value_idx, _value_shape));

                if (q_len == 1 && seq_idx > 0) continue;
                dram_query_addrs.insert(make_address(query_idx, _query_shape));
                dram_output_addrs.insert(make_address(output_idx, _query_shape)); // Used query_shape intentionally
            }
        }
        // -- load --
        // MOVIN query, key, value
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_q_ofs,
            .size = (uint32_t)dram_query_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_query_addrs.begin(), dram_query_addrs.end()),
            .operand_id = _INPUT_OPERAND,  // query
        });
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_k_ofs,
            .size = (uint32_t)dram_key_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_key_addrs.begin(), dram_key_addrs.end()),
            .operand_id = _INPUT_OPERAND + 1,  // key
        });
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_v_ofs,
            .size = (uint32_t)dram_value_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_value_addrs.begin(), dram_value_addrs.end()),
            .operand_id = _INPUT_OPERAND + 2,  // value
        });
        // -- compute --
        // GEMM (q*k -> l)
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = q_len * seq_len,
            .src_addrs = std::vector<addr_type>{sram_q_ofs, sram_k_ofs},

            .tile_m = seq_len,
            .tile_k = _dk,
            .tile_n = q_len,
        });
        // Softmax (l -> l)

        tile.instructions.push_back(Instruction{
            .opcode = Opcode::SOFTMAX,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = seq_len * _config.precision,
            .src_addrs = std::vector<addr_type>{sram_l_ofs},
            .tile_m = q_len,
            .src_from_accum = true,
        });

        // [ ] change output offset
        // GEMM (l*v -> acc)
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_l_ofs,
            .size = q_len * _dk * _config.precision / _config.dram_req_size,
            .compute_size = q_len * _dk,
            .src_addrs = std::vector<addr_type>{sram_l_ofs, sram_v_ofs},

            .tile_m = _dk,
            .tile_k = seq_len,
            .tile_n = q_len,
            .src_from_accum = true,
        });

        // MOVOUT
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_l_ofs,
            .size = (uint32_t)dram_output_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_output_addrs.begin(), dram_output_addrs.end()),
            .operand_id = _OUTPUT_OPERAND,
        });
    }
}

void Attention::calculate_loops() {
    for (int i = 0; i < _batch_size; i++) {
        uint32_t q_len = _q_len;
        uint32_t seq_len = _seq;

        uint32_t total_spad_size_per_head = 2*seq_len*_dk + q_len*_dk;
        uint32_t total_acc_size_per_head = seq_len*q_len;
        // query: q_len * _dk
        // key: seq_len * _dk
        // value: seq_len * _dk
        // query_key: seq_len * q_len
        // out: q_len * _dk
        total_acc_size_per_head *= _config.precision;
        total_spad_size_per_head *= _config.precision;

        uint32_t spad_capacity = _config.spad_size KB / 2;  // unit: byte
        uint32_t acc_spad_capacity = _config.accum_spad_size KB / 2;

        uint32_t heads_per_tile = std::min(spad_capacity / total_spad_size_per_head,
                                            acc_spad_capacity/ total_acc_size_per_head);
        if (heads_per_tile > _nh) heads_per_tile = _nh;
        assert (heads_per_tile >= 1);
        spdlog::info("[Fused Attention] ({}) heads_per_tile: {}", i, heads_per_tile);
        spdlog::info("[Fused Attention] q_len: {}, seq_len: {}, dk: {}", q_len, seq_len, _dk);
        spdlog::info("[Fused Attention] spad capacity: 0x{:x}, acc spad capacity: 0x{:x}, " \
            "one head spad size: 0x{:x}, acc spad size: 0x{:x}",
            spad_capacity, acc_spad_capacity, total_spad_size_per_head, total_acc_size_per_head);
        if (heads_per_tile <=0) {
            spdlog::error("Spad capacity is too small!");
            exit(EXIT_FAILURE);
        }
        _heads_per_tile.push_back(heads_per_tile);
    }
}

addr_type Attention::make_address(std::vector<uint32_t> index, std::vector<uint32_t> dims) {
    assert(index.size() == 3 && dims.size() == 3);
    addr_type address;

    address  = index[0] * (dims[1] * dims[2]) + index[1] * (dims[2]) + index[2];
    address = _config.align_address(address * _config.precision);
    return address;
}

uint32_t Attention::sram_size_needed() { return 0; }