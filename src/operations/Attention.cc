#include "Attention.h"

Attention::Attention(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {
    
    for (auto attribute : node_proto.attribute()) {
        if (attribute.name() == "num_heads") {
            _nh = attribute.i();
        }
    }
    _batch_size = 1;
    _dk = config.model_n_embd;
    _seq = config.model_seq_len;

    calculate_loops();
}

/**
 * MHA not including QKV generation and projection layer.
 * Takes batched query, key, value tensors.
 * q,k,v shape : (nh,{1,l},dk) / (nh,{l,l+1},dk) / (nh,{l,l+1},dk)
 * a shape : (nh,q_len,dk)
 *  -> fixed to a shape : (q_len,E)
 * inputs: for batch size n,
 *  (q1,...,qn,k1,...,kn,v1,...,vn)
 * output:
 *  (a1,...,an)
std::vector<Ptr<BTensor>> Attention::get_outputs(std::vector<Ptr<BTensor>> inputs) {
    set_as_parent_tensor(inputs);

    _inputs = inputs;

    _batch_size = inputs.size() / 3;
    uint32_t i = 0;

    // * q = seq_len or 1
    // query [h, q, dk]
    // key [h, dk, seq_len]
    // value [h, seq_len, dk]

    for (auto tensor : inputs) {
        if (i < _batch_size) {
            _query.push_back(std::static_pointer_cast<NPUTensor>(tensor));
        } else if (i < _batch_size * 2) {
            _key.push_back(std::static_pointer_cast<NPUTensor>(tensor));
        } else {
            _value.push_back(std::static_pointer_cast<NPUTensor>(tensor));
        }
        i++;
    }

    _outputs.resize(_batch_size);

    _nh = _query[0]->get_dims()[0];
    _dk = _query[0]->get_dims()[2];

    ast(_nh == Config::global_config.model_n_head / Config::global_config.n_tp);
    ast(_dk == Config::global_config.model_n_embd / Config::global_config.model_n_head);

    for (int i = 0; i < _batch_size; ++i) {
        auto q = _query[i];  // [h, q, dk]
        auto k = _key[i];    // [h, dk, seq_len]
        auto v = _value[i];  // [h, seq_len, dk]

        // seq_len of key == seq_len of value
        spdlog::info("Attention Q:{}, K:{}, V:{}", q->get_dims(), k->get_dims(), v->get_dims());
        assert(k->get_dims()[2] == v->get_dims()[1]);
        uint32_t q_len = q->get_dims()[1];  // seq_len or 1
        assert(q_len == 1 || q_len == k->get_dims()[2]);
        // xxx
        std::vector<uint32_t> mha_output_dim{_nh, q_len, _dk};
        // std::vector<uint32_t> mha_output_dim{q_len, _dk * _nh};
        spdlog::info("Attention Q:{}, K:{}, V:{}", q->get_dims(), k->get_dims(), v->get_dims());
        _outputs[i] = std::make_shared<NPUTensor>(_name + "_output", mha_output_dim,
                                                  NPUTensorBufType::ACT, false);
    }

    // todo tiling and instruction initialization.
    calculate_loops();
    initialize_tiles();

    spdlog::info("Attention (batch size): {}", _batch_size);

    return _outputs;
}
*/
void Attention::initialize_tiles(MappingTable mapping_table) {
    for (int req_idx = 0; req_idx < _batch_size; req_idx++) {
        int heads_per_tile = _heads_per_tile[req_idx];
        int head_idx = 0;

        auto tile = Tile{
            .status = Tile::Status::INITIALIZED,
            .optype = get_name(),
            .layer_id = _id,
            //.K = 0,
            .accum = false,
        };
        /* dummy mapping */
        Mapping mapping;
        initialize_instructions(tile, mapping, 0, heads_per_tile);

        _tiles.push_back(tile);
    }
}

// 일단 한 tile에는 최대 하나의 request만 있는 경우부터.
void Attention::initialize_instructions(Tile &tile, Mapping mapping, int head_idx, int num_heads) {
    // req_idx in batch
    // head_idx # start idx
    // num_heads
    int q_len = _seq;
    int seq_len = _seq;

    addr_type sram_query_base = SPAD_BASE;
    addr_type sram_key_base = sram_query_base + q_len * _dk * num_heads * _config.precision;
    addr_type sram_value_base = sram_key_base + _dk * seq_len * num_heads * _config.precision;
    addr_type sram_logit_base = ACCUM_SPAD_BASE;  // for logits
    addr_type sram_accumulation_base =
        sram_logit_base + q_len * seq_len * num_heads * _config.precision;

    for (int h_ofs = 0; h_ofs < num_heads; h_ofs++) {
        int h_idx = head_idx + h_ofs;

        addr_type sram_q_ofs = sram_query_base + h_ofs * (q_len * _dk) * _config.precision;
        addr_type sram_k_ofs = sram_key_base + h_ofs * (_dk * seq_len) * _config.precision;
        addr_type sram_v_ofs = sram_value_base + h_ofs * (_dk * seq_len) * _config.precision;
        addr_type sram_l_ofs = sram_logit_base + h_ofs * (q_len * seq_len) * _config.precision;
        addr_type sram_acc_ofs = sram_accumulation_base + h_ofs * (q_len * _dk) * _config.precision;

        std::vector<addr_type> dram_query_addrs;  // = _query[req_idx]->get_all_addrs();
        std::vector<addr_type> dram_key_addrs;    // = _key[req_idx]->get_all_addrs();
        std::vector<addr_type> dram_value_addrs;

        for (int i = 0; i < _dk; i++) {
            for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
                // key:  h, d_k, seq_len
                dram_key_addrs.push_back((h_idx*(_dk*_seq)+i*_seq+seq_idx) * _config.precision);

                // value: h, seq_len, d_k
                dram_value_addrs.push_back((h_idx*(_dk*_seq)+seq_idx*_dk+i) * _config.precision);

                if (q_len == 1 && seq_idx > 0) continue;
                dram_query_addrs.push_back((h_idx*(_dk*_seq)+seq_idx*_dk+i) * _config.precision);
            }

            // dram_key_addrs.push_back(_key[req_idx]->get_addr(std::vector<uint32_t>{h_idx, i}));
            // dram_value_addrs.push_back(_value[req_idx]->get_addr(std::vector<uint32_t>{h_idx,
            // i}));
        }
        // spdlog::info("dram_query_addrs.size(): {}", dram_query_addrs.size());
        // spdlog::info("dram_key_addrs.size(): {}", dram_key_addrs.size());
        // spdlog::info("dram_value_addrs.size(): {}", dram_key_addrs.size());

        assert(dram_query_addrs.size() == q_len * _dk);
        assert(dram_key_addrs.size() == seq_len * _dk);
        assert(dram_value_addrs.size() == seq_len * _dk);

        // -- load --
        // MOVIN query, key, value
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_q_ofs,
            .size = (q_len * _dk) * _config.precision,
            .src_addrs = std::move(dram_query_addrs),
            .operand_id = _INPUT_OPERAND,  // query
        });
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_k_ofs,
            .size = (seq_len * _dk) * _config.precision,
            .src_addrs = std::move(dram_key_addrs),
            .operand_id = _INPUT_OPERAND + 1,  // key
        });
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_v_ofs,
            .size = (seq_len * _dk) * _config.precision,
            .src_addrs = std::move(dram_value_addrs),
            .operand_id = _INPUT_OPERAND + 2,  // value
        });

        // -- compute --
        // GEMM (q*k -> l)
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len,
            .src_addrs = std::vector<addr_type>{sram_q_ofs, sram_k_ofs},

            //.tile_m = seq_len,
            //.tile_k = _dk,
            //.tile_n = q_len,
        });
        // Softmax (l -> l)

        tile.instructions.push_back(Instruction{
            .opcode = Opcode::SOFTMAX,
            .dest_addr = sram_acc_ofs,
            .size = q_len * seq_len,
            .src_addrs = std::vector<addr_type>{sram_l_ofs},
            //.src_from_accum = true,
        });

        // [ ] change output offset
        addr_type output_ofs = sram_acc_ofs + q_len * _dk * _config.precision;
        // GEMM (l*v -> acc)
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = output_ofs,
            .size = q_len * _dk,
            .src_addrs = std::vector<addr_type>{sram_acc_ofs, sram_v_ofs},

            //.tile_m = _dk,
            //.tile_k = seq_len,
            //.tile_n = q_len,
            //.src_from_accum = true,
        });

        // MOVOUT
        tile.instructions.push_back(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = output_ofs,
            .size = q_len * _dk * _config.precision,
            .src_addrs = std::vector<addr_type>{output_ofs},
            //std::move(std::static_pointer_cast<NPUTensor>(_outputs[req_idx])
            //                           ->_inners[h_idx]
            //                           ->get_all_addrs()),
            .operand_id = _OUTPUT_OPERAND,
        });
    }
}

void Attention::calculate_loops() {
    // todo: tiling!
    // different sequence length for each request in a batch...
    // request마다 head attention을 몇개씩 batching할 것인지. 결정?
    /*
    for (req in batch)
        q = _query[i] // [h, q, dk] #여기서 q는 seq_len이거나 1임
        k = _key[i] // [h, dk, seq_len]
        v = _value[i] // [h, seq_len, dk]

        We need q, k, v for a head attention computation
        q : q*dk
        k : dk*seq_len
        v : dk*seq_len
        intermediate : seq_len*q
        output : dk*q

        total size per head :
            2*q*dk + 2*dk*seq_len + seq_len*q

        - [] 각 request마다 몇개의 head computation을 한 tile로 묶을 수 있는지 계산 ->
    _heads_per_tile에 저장.
        - [] initialize_instructions()에서 tile 하나 하나 계산.
    */

    for (int i = 0; i < _batch_size; i++) {
        uint32_t q_len = _seq;
        uint32_t seq_len = _seq;

        uint32_t total_size_per_head = 2 * q_len * _dk + 2 * _dk * seq_len + seq_len * q_len;
        total_size_per_head *= _config.precision;  // unit: byte

        uint32_t sram_capacity = _config.spad_size KB / 2;  // unit: byte

        uint32_t heads_per_tile = sram_capacity / total_size_per_head;
        assert (heads_per_tile >= 1);
        if (heads_per_tile > _nh) heads_per_tile = _nh;


        spdlog::info("({}) heads_per_tile: {}", i, heads_per_tile);
        spdlog::info("q_len: {}, seq_len: {}, dk: {}", q_len, seq_len, _dk);
        spdlog::info("sram capacity: {}, one head size: {}", sram_capacity, total_size_per_head);

        _heads_per_tile.push_back(heads_per_tile);
    }
}

uint32_t Attention::sram_size_needed() { return 0; }