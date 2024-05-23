#include "KVCacheConcat.h"
#include "../Model.h"

KVCacheConcat::KVCacheConcat(SimulationConfig config, Model* model,
                             onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {
  spdlog::error("KVCacheConcat: Not implemented");
  throw std::runtime_error("KVCacheConcat: Not implemented");
}

KVCacheConcat::KVCacheConcat(const KVCacheConcat& src) : Operation(src) {
  spdlog::error("KVCacheConcat: Not implemented");
  throw std::runtime_error("KVCacheConcat: Not implemented");
}

KVCacheConcat::KVCacheConcat(SimulationConfig config, Model* model,
                             std::string name,
                             std::map<std::string, std::string>& attributes)
    : Operation(config, model, name, attributes) {
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
  _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED,
                      .optype = "KVCacheConcat",
                      .layer_id = _id,
                      .skip = true}));
  // TODO:implemt
}