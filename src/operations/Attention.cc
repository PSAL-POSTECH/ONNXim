#include "Attention.h"
#include "../Model.h"
#include "../Tensor.h"
#include "GemmWS.h"
#include "Softmax.h"

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
    _projection_output_shape = std::vector<uint32_t>{_batch_size, _q_len, _weight_shape[1]/3};
    spdlog::debug("Fused attention: input shape: [{}, {}, {}]", _input_shape.at(0), _input_shape.at(1), _input_shape.at(2));
    spdlog::debug("Fused attention: output shape: [{}, {}, {}]", _output_shape.at(0), _output_shape.at(1), _output_shape.at(2));
    spdlog::debug("Fused attention: query shape: [{}, {}, {}]", _query_shape.at(0), _query_shape.at(1), _query_shape.at(2));
    spdlog::debug("Fused attention: key shape: [{}, {}, {}]", _key_shape.at(0), _key_shape.at(1), _key_shape.at(2));
    spdlog::debug("Fused attention: value shape: [{}, {}, {}]", _value_shape.at(0), _value_shape.at(1), _value_shape.at(2));

    Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(0));
    if (pre_defind_tensor == nullptr) {
        std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
            _id, node_proto.output(0), _output_shape, _config.precision, false);
            _outputs.push_back(output_tensor.get()->get_id());
        _model->add_tensor(std::move(output_tensor));
    } else {
        pre_defind_tensor->redefine_tensor(_id, _output_shape);
    }
    calculate_loops();
}

Attention::Attention(SimulationConfig config, Model* model, 
        std::string name, std::map<std::string, std::string>& attributes)
    :Operation(config, model, name, attributes) {
    //TODO: implement this
    uint32_t num_tokens = std::stoi(get_attribute("num_tokens"));
    uint32_t num_heads = std::stoi(get_attribute("num_heads"));
    uint32_t num_kv_heads = std::stoi(get_attribute("num_kv_heads"));
    uint32_t hidden_dim = std::stoi(get_attribute("hidden_size"));
    uint32_t head_size = hidden_dim / num_heads;
    std::vector<uint32_t> out_dim =  {num_tokens, hidden_dim};
    auto output_tensor = std::make_unique<Tensor> (
        _id, name_gen(_name, "output"), out_dim, _config.precision, false);
    _outputs.push_back(output_tensor.get()->get_id());
    _model->add_tensor(std::move(output_tensor));
}

void Attention::initialize_tiles(MappingTable& mapping_table) {
    _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED,
                      .optype = "Attention",
                      .layer_id = _id,
                      .skip = true}));
    return; //TODO: fixme
    /* Check using fusion */
    if (!use_fused) {
        initialize_non_fused_tiles(mapping_table);
        return;
    }

    /* Create linear node and tensors */
    uint32_t fused_op_id = 0;
    _projection_node = new GemmWS(_config, mapping_table, _input_shape, _weight_shape, _liner_output_shape);
    std::unique_ptr<Tensor> key_projection = std::make_unique<Tensor>(
        _id, "", _projection_output_shape, _config.precision, false);
    std::unique_ptr<Tensor> query_projection = std::make_unique<Tensor>(
        _id, "", _projection_output_shape, _config.precision, false);
    std::unique_ptr<Tensor> value_projection = std::make_unique<Tensor>(
       _id, "", _projection_output_shape, _config.precision, false);

    /* Link tensors to linear node */
    _projection_node->set_model(_model);
    _projection_node->add_input(_inputs.at(0));
    _projection_node->add_input(_inputs.at(1));
    _projection_node->add_input(_inputs.at(2));
    _projection_node->add_output(key_projection.get()->get_id());
    _projection_node->add_output(query_projection.get()->get_id());
    _projection_node->add_output(value_projection.get()->get_id());
    get_input(0)->add_child_node(_projection_node);
    key_projection->add_child_node(this);
    query_projection->add_child_node(this);
    value_projection->add_child_node(this);

    /* Link key query value to attention node */
    _key_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(key_projection.get()->get_id());
    _query_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(query_projection.get()->get_id());
    _value_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(value_projection.get()->get_id());

    /* Register tensor */
    _model->add_tensor(std::move(key_projection));
    _model->add_tensor(std::move(query_projection));
    _model->add_tensor(std::move(value_projection));

    /* Fused Attention body */
    for (int req_idx = 0; req_idx < _batch_size; req_idx++) {
        int heads_per_tile = _heads_per_tile[req_idx];
        for (int head_off=0; head_off<_nh; head_off+=heads_per_tile) {
            uint32_t remain_heads = std::min(_nh-head_off, (uint32_t)heads_per_tile);
            std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
                .status = Tile::Status::INITIALIZED,
                .optype = get_name(),
                .layer_id = _id,
                .fused_op_id = fused_op_id++,
                //.K = 0,
                .accum = false,
            });
            /* dummy mapping */
            Mapping mapping;
            _tiles.push_back(std::move(tile));
            initialize_instructions(_tiles.back().get(), mapping, head_off, heads_per_tile);
        }
    }
}

// 일단 한 tile에는 최대 하나의 request만 있는 경우부터.
void Attention::initialize_instructions(Tile* tile, Mapping mapping, int head_idx, int num_heads) {
    // head_idx # start idx
    // num_heads
    uint32_t q_len = _q_len;
    uint32_t seq_len = _seq;

    addr_type sram_query_base = SPAD_BASE;
    addr_type sram_key_base = sram_query_base + q_len * _dk * num_heads * _config.precision;
    addr_type sram_value_base = sram_key_base + _dk * seq_len * num_heads * _config.precision;
    addr_type sram_logit_base = ACCUM_SPAD_BASE;  // for logits

    addr_type key_addr = get_operand_addr(_key_projection_id);
    addr_type query_addr = get_operand_addr(_query_projection_id);
    addr_type value_addr = get_operand_addr(_value_projection_id);
    addr_type ouput_addr = get_operand_addr(_OUTPUT_OPERAND);
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
                std::vector<uint32_t> key_idx =   {(uint32_t)(h_idx), (uint32_t)i, (uint32_t)seq_idx};
                std::vector<uint32_t> value_idx = {(uint32_t)(h_idx), (uint32_t)seq_idx, (uint32_t)i};
                std::vector<uint32_t> output_idx = {(uint32_t)(h_idx), (uint32_t)seq_idx, (uint32_t)i};

                dram_key_addrs.insert(key_addr + make_address(key_idx, _key_shape));
                dram_value_addrs.insert(value_addr + make_address(value_idx, _value_shape));

                if (q_len == 1 && seq_idx > 0) continue;
                dram_query_addrs.insert(query_addr + make_address(query_idx, _query_shape));
                dram_output_addrs.insert(ouput_addr + make_address(output_idx, _query_shape)); // Used query_shape intentionally
            }
        }
        // -- load --
        // MOVIN query, key, value
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_q_ofs,
            .size = (uint32_t)dram_query_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_query_addrs.begin(), dram_query_addrs.end()),
            .operand_id = _INPUT_OPERAND,  // query
        }));
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_k_ofs,
            .size = (uint32_t)dram_key_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_key_addrs.begin(), dram_key_addrs.end()),
            .operand_id = _INPUT_OPERAND + 1,  // key
        }));
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_v_ofs,
            .size = (uint32_t)dram_value_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_value_addrs.begin(), dram_value_addrs.end()),
            .operand_id = _INPUT_OPERAND + 2,  // value
        }));
        // -- compute --
        // GEMM (q*k -> l)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = std::min((int(q_len/_config.core_height)), 1) * seq_len,
            .src_addrs = std::vector<addr_type>{sram_q_ofs, sram_k_ofs},

            .tile_m = seq_len,
            .tile_k = _dk,
            .tile_n = q_len,
        }));
        // Softmax (l -> l)

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SOFTMAX,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = seq_len * _config.precision,
            .src_addrs = std::vector<addr_type>{sram_l_ofs},
            .tile_m = q_len,
            .src_from_accum = true,
        }));

        // [ ] change output offset
        // GEMM (l*v -> acc)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_l_ofs,
            .size = q_len * _dk * _config.precision / _config.dram_req_size,
            .compute_size = q_len * _dk,
            .src_addrs = std::vector<addr_type>{sram_l_ofs, sram_v_ofs},

            .tile_m = _dk,
            .tile_k = seq_len,
            .tile_n = q_len,
            .src_from_accum = true,
        }));

        // MOVOUT
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_l_ofs,
            .size = (uint32_t)dram_output_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_output_addrs.begin(), dram_output_addrs.end()),
            .operand_id = _OUTPUT_OPERAND,
        }));
    }
}

void Attention::initialize_non_fused_tiles(MappingTable& mapping_table) {
    /* Create linear node and tensors */
    uint32_t fused_op_id = 0;
    _projection_node = new GemmWS(_config, mapping_table, _input_shape, _weight_shape, _liner_output_shape);
    std::unique_ptr<Tensor> key_projection = std::make_unique<Tensor>(
        _id, "", _projection_output_shape, _config.precision, true);
    std::unique_ptr<Tensor> query_projection = std::make_unique<Tensor>(
        _id, "", _projection_output_shape, _config.precision, true);
    std::unique_ptr<Tensor> value_projection = std::make_unique<Tensor>(
       _id, "", _projection_output_shape, _config.precision, true);
    _model->add_tensor(std::move(key_projection));
    _model->add_tensor(std::move(query_projection));
    _model->add_tensor(std::move(value_projection));

    /* Link tensors to linear node */
    _projection_node->set_model(_model);
    _projection_node->add_input(_inputs.at(0));
    _projection_node->add_input(_inputs.at(1));
    _projection_node->add_output(key_projection.get()->get_id());
    _projection_node->add_output(query_projection.get()->get_id());
    _projection_node->add_output(value_projection.get()->get_id());

    /* Link key query value to attention node */
    _key_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(key_projection.get()->get_id());
    _query_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(query_projection.get()->get_id());
    _value_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(value_projection.get()->get_id());

    /* Initilize tiles */
    _projection_node->has_bias = false;
    _projection_node->initialize_tiles(mapping_table);
    std::deque<std::unique_ptr<Tile>>& tiles = _projection_node->get_tiles();
    for (const auto& tile : tiles) {
        tile->layer_id = _id;
        tile->fused_op_id = fused_op_id;
    }
    _tiles.insert(
        _tiles.end(),
        std::make_move_iterator(_projection_node->get_tiles().begin()),
        std::make_move_iterator(_projection_node->get_tiles().end())
    );
    _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::BAR, .layer_id = _id}));
    fused_op_id++;
    std::vector<uint32_t> single_head_query_shape = std::vector<uint32_t>{_q_len, _dk};
    std::vector<uint32_t> single_head_key_shape = std::vector<uint32_t>{_dk, _seq};
    std::vector<uint32_t> single_head_value_shape = std::vector<uint32_t>{_seq, _dk};
    std::vector<uint32_t> single_output_shape = std::vector<uint32_t>{_q_len, _dk};
    std::vector<uint32_t> query_key_shape = std::vector<uint32_t>{_q_len, _seq};

    /* Fused Attention body */
    for (int req_idx = 0; req_idx < _batch_size; req_idx++) {
        for (int head_off=0; head_off<_nh; head_off++) {
            /* Key query matmul */
            GemmWS key_query = GemmWS(_config, mapping_table, single_head_query_shape, single_head_key_shape, query_key_shape);
            /* Todo. dram addr */
            key_query.has_bias = false;
            key_query.initialize_tiles(mapping_table);
            std::deque<std::unique_ptr<Tile>>& key_query_tiles = key_query.get_tiles();
            for (const auto& tile : key_query_tiles) {
                tile->layer_id = _id;
                tile->fused_op_id = fused_op_id;
            }
            _tiles.insert(
                _tiles.end(),
                std::make_move_iterator(key_query.get_tiles().begin()),
                std::make_move_iterator(key_query.get_tiles().end())
            );
            _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::BAR, .layer_id = _id}));
            fused_op_id++;

            /* Softmax */
            Softmax attention_score = Softmax(_config, mapping_table, query_key_shape);
            /* Todo. dram addr */
            attention_score.initialize_tiles(mapping_table);
            std::deque<std::unique_ptr<Tile>>& attention_score_tiles = key_query.get_tiles();
            for (const auto& tile : attention_score_tiles) {
                tile->layer_id = _id;
            }
            _tiles.insert(
                _tiles.end(),
                std::make_move_iterator(key_query.get_tiles().begin()),
                std::make_move_iterator(key_query.get_tiles().end())
            );
            _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::BAR, .layer_id = _id}));

            /* attention x value */
            GemmWS attention = GemmWS(_config, mapping_table, query_key_shape, single_head_value_shape, single_output_shape);
            /* Todo. dram addr */
            attention.has_bias = false;
            attention.initialize_tiles(mapping_table);
            std::deque<std::unique_ptr<Tile>>& attention_tiles = attention.get_tiles();
            for (const auto& tile : attention_tiles) {
                tile->layer_id = _id;
                tile->fused_op_id = fused_op_id;
            }
            _tiles.insert(
                _tiles.end(),
                std::make_move_iterator(attention.get_tiles().begin()),
                std::make_move_iterator(attention.get_tiles().end())
            );
            _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::BAR, .layer_id = _id}));
            fused_op_id++;
        }
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

        if (heads_per_tile <= 0) {
            use_fused = false;
            spdlog::info("[Attention] Use non fusion attention!");
            break;
        }

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