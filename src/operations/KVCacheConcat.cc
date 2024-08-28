#include "KVCacheConcat.h"
#include "../Model.h"
// const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
// const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
//                                       // head_size/x, block_size, x]
// const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
//                                       // head_size, block_size]
// const int num_kv_heads,               // [num_heads]
KVCacheConcat::KVCacheConcat(SimulationConfig config, Model* model,
                             onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  spdlog::error("KVCacheConcat: Not implemented");
  throw std::runtime_error("KVCacheConcat: Not implemented");
}

KVCacheConcat::KVCacheConcat(const KVCacheConcat& src) : Operation(src) {
  spdlog::error("KVCacheConcat: Not implemented");
  throw std::runtime_error("KVCacheConcat: Not implemented");
}

KVCacheConcat::KVCacheConcat(SimulationConfig config, Model* model,
                             std::string name,
                             std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _input_token_lengths = parse_dims(get_attribute("input_token_lengths"));
  _num_kv_heads = std::stoi(get_attribute("num_kv_heads"));
  _num_attention_heads = std::stoi(get_attribute("num_heads"));
  _hidden_size = std::stoi(get_attribute("hidden_size"));
  _num_batches = _input_token_lengths.size();
  _cache_dim = _hidden_size / _num_attention_heads * _num_kv_heads;
  
  spdlog::debug("[KVCacheConcat] input_token_lengths: {}",
                _input_token_lengths);
  for (int batch = 0; batch < _num_batches; batch++) {
    std::vector<uint32_t> query_dim = {_input_token_lengths[batch], _hidden_size};
    auto query_out = std::make_unique<Tensor>(
        _id, name_gen(std::to_string(_id), "QueryOut", std::to_string(batch)),
        query_dim, _config.precision, false);
    //make temporal tensor for key and value
    auto key_out = std::make_unique<Tensor>(
        _id, name_gen(std::to_string(_id), "KeyOut", std::to_string(batch)),
        _config.precision);
    auto value_out = std::make_unique<Tensor>(
        _id, name_gen(std::to_string(_id), "ValueOut", std::to_string(batch)),
        _config.precision);
    _outputs.push_back(query_out.get()->get_id());
    _model->add_tensor(std::move(query_out));
    _outputs.push_back(key_out.get()->get_id());
    _model->add_tensor(std::move(key_out));
    _outputs.push_back(value_out.get()->get_id());
    _model->add_tensor(std::move(value_out));
  }
}

void KVCacheConcat::initialize_tiles(MappingTable& mapping_table) {
  auto qkv_out_tensor = _model->get_tensor(_inputs[0]);
  for(int batch = 0; batch < _num_batches; batch++){
    uint32_t key_tensor_id = _outputs[batch * 3 + 1];
    uint32_t value_tensor_id = _outputs[batch * 3 + 2];
    auto key_cache = _model->get_tensor(_inputs[batch * 2 + 1]);
    auto key_dims = key_cache->get_dims();
    key_dims[0] = key_dims[0] + _input_token_lengths[batch];
    _model->get_tensor(key_tensor_id)->define_tensor(key_cache->get_address(), key_dims); 
    auto value_cache = _model->get_tensor(_inputs[batch * 2 + 2]);
    auto value_dims = value_cache->get_dims();
    value_dims[0] = value_dims[0] + _input_token_lengths[batch];
    _model->get_tensor(value_tensor_id)->define_tensor(value_cache->get_address(), value_dims);
  }
  calculate_loops();
  for(int outter = 0; outter < _outter_loops; outter++) {
    _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED,
                      .optype = "KVCacheConcat",
                      .layer_id = _id,
                      .skip = true}));
    initialize_instructions(_tiles.back().get(), outter);
  }
}

void KVCacheConcat::calculate_loops() {
  uint32_t per_token_size = _config.precision * ( _cache_dim * 2 + _hidden_size);
  _outter_loops =  ceil_div(get_input(0)->get_size() ,(_config.core_config[target_core].spad_size KB/ 2));
  _inner_loops = ceil_div(_config.core_config[target_core].spad_size KB/2, per_token_size);
  spdlog::debug("[KVCacheConcat] number of tiles: {}", _outter_loops);
}

void KVCacheConcat::initialize_instructions(Tile* tile, uint32_t idx) {
  uint32_t per_token_size = _config.precision * ( _cache_dim * 2 + _hidden_size);
  std::set<addr_type> movin_addresses;
  std::set<addr_type> query_out_address;
  std::vector<std::set<addr_type>> key_out_addresses;
  std::vector<std::set<addr_type>> value_out_addresses;
  key_out_addresses.resize(_num_batches);
  value_out_addresses.resize(_num_batches);

  int currenet_batch = 0;
  int current_index = 0;
  for(int inner = 0; _inner_loops; inner++) {
    int token_id = idx * _inner_loops + inner;
    if(token_id >= get_input(0)->get_dims()[0]) {
      break;
    }
    addr_type qkv_out_addr = get_input(0)->get_address() + token_id * per_token_size;
    addr_type query_out_addr = get_output(0)->get_address() + token_id * _hidden_size * _config.precision;
    for(addr_type offset = 0; offset < per_token_size; offset += _config.dram_req_size) {
      movin_addresses.insert(_config.align_address(qkv_out_addr + offset));
    }
    for(addr_type offset = 0; offset < _hidden_size * _config.precision; offset += _config.dram_req_size) {
      query_out_address.insert(_config.align_address(query_out_addr + offset));
    }
    auto key_cache_tensor = get_output(currenet_batch * 3 + 1);
    addr_type key_out_address = key_cache_tensor->get_address() + key_cache_tensor->get_size() + 
      current_index * _cache_dim * _config.precision;
    addr_type value_out_address = key_cache_tensor->get_address() + key_cache_tensor->get_size() + 
      current_index * _cache_dim * _config.precision;
    for(addr_type offset = 0; offset < _cache_dim * _config.precision; offset += _config.dram_req_size) {
      key_out_addresses[currenet_batch].insert(_config.align_address(key_out_address + offset));
      value_out_addresses[currenet_batch].insert(_config.align_address(value_out_address + offset));
    }

    if(current_index >= _input_token_lengths[currenet_batch]) {
      currenet_batch++;
      current_index = 0;
    }
    current_index++;
  }
  
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
    .opcode = Opcode::MOVIN,
    .dest_addr = SPAD_BASE,
    .size = (uint32_t)movin_addresses.size(),
    .src_addrs = std::vector<addr_type>(movin_addresses.begin(), movin_addresses.end()),
    .operand_id = _INPUT_OPERAND
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
    .opcode = Opcode::MOVOUT,
    .dest_addr = SPAD_BASE,
    .size = (uint32_t)query_out_address.size(),
    .src_addrs = std::vector<addr_type>(query_out_address.begin(), query_out_address.end()),
    .operand_id = _OUTPUT_OPERAND
  }));
  for(int batch = 0; batch < _num_batches; batch++) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = SPAD_BASE,
      .size = (uint32_t)key_out_addresses[batch].size(),
      .src_addrs = std::vector<addr_type>(key_out_addresses[batch].begin(), key_out_addresses[batch].end()),
      .operand_id = _OUTPUT_OPERAND
    }));
  }
  for(int batch = 0; batch < _num_batches; batch++){
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = SPAD_BASE,
      .size = (uint32_t)value_out_addresses[batch].size(),
      .src_addrs = std::vector<addr_type>(value_out_addresses[batch].begin(), value_out_addresses[batch].end()),
      .operand_id = _OUTPUT_OPERAND
    }));
  }
}