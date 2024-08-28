#include "LanguageModel.h"
#include "../operations/Operation.h"
#include "../operations/Attention.h"
#include "../operations/Concat.h"
#include "../operations/GemmWS.h"
#include "../operations/BiasAct.h"
#include "../operations/KVCacheConcat.h"
#include "../operations/SkipLayerNorm.h"

namespace BlockType {
std::string Attention = "attn";
std::string FeedForward = "ffn";
}  // namespace BlockType

namespace OperationType {
std::string LayerNorm = "ln";
std::string QKVGen = "QKVgen";
std::string Projection = "proj";
std::string FullyConnected1 = "fc1";
std::string FullyConnected2 = "fc2";
std::string LmHead = "lmhead";
std::string Act = "act";
std::string KVCacheConcat = "KVccat";
std::string AttentionConcat = "Atccat";
std::string Attention = "Attention";
}  // namespace OperationType

namespace ParameterType {
std::string Weight = "weight";
std::string Bias = "bias";
}  // namespace ParameterType


LanguageModel::LanguageModel(json llm_config, SimulationConfig config, std::string name) : Model(llm_config, config, name) {
  //Constructor for weight initialization
  //Not used for actual simulation
  _num_batch = 0;
  _run_single_layer = false;
  _tensor_parallel_size = 1;
  _pipeline_parallel_size = 1;
  if(llm_config.contains("run_single_layer")) {
    _run_single_layer = llm_config["run_single_layer"];
  }
  if(_model_config.contains("tensor_parallel_size")) {
    _tensor_parallel_size = _model_config["tensor_parallel_size"];
  }
  if(_model_config.contains("pipeline_parallel_size")) {
    _pipeline_parallel_size = _model_config["pipeline_parallel_size"];
  }
  _num_layers = llm_config["num_hidden_layers"];
  _num_layers /= _pipeline_parallel_size;
  _num_sim_layers = _run_single_layer ? 1 : _num_layers;
  _hidden_size = llm_config["hidden_size"];
  _num_kv_heads = llm_config["num_kv_heads"];
  _num_heads = llm_config["num_attention_heads"];
  _qkv_out_dim = _hidden_size / _num_heads * _num_kv_heads * 2 + _hidden_size;
  _qkv_out_dim /= _tensor_parallel_size;
  _num_kv_heads /= _tensor_parallel_size;
  _num_heads /= _tensor_parallel_size;
  _proj_in_dim = _hidden_size;
  _proj_in_dim /= _tensor_parallel_size;
  _intermediate_size = llm_config["intermediate_size"];
  _intermediate_size /= _tensor_parallel_size;

  _ffn1_out_dim = _intermediate_size;
  _llama_mlp = _model_config["ffn_type"] == "llama";
  if(_llama_mlp) {
    _ffn1_out_dim *= 2;
  }
}

std::unique_ptr<LanguageModel> LanguageModel::generate_model(std::vector<LangInput>& reqs) {
  std::unique_ptr<LanguageModel> model = std::make_unique<LanguageModel>(_model_config, _config, _name);
  model->_root_node_id = _root_node_id;
  model->_reqs = reqs;
  model->_wgt_map = _wgt_map;
  model->_wgt_size = _wgt_size;
  model->_num_batch = reqs.size();
  //Load KV Cache
  model->_key_cache_tensor_ids.resize(reqs.size());
  model->_value_cache_tensor_ids.resize(reqs.size());
  for(int b = 0; b < reqs.size(); b++) {
    model->_key_cache_tensor_ids[b].resize(_num_sim_layers);
    model->_value_cache_tensor_ids[b].resize(_num_sim_layers);
    for(int l = 0; l < _num_sim_layers; l++) {
      auto key_cache = std::make_unique<Tensor>(*reqs[b].key_cache[l]);
      key_cache->set_produced();
      model->_key_cache_tensor_ids[b][l] = key_cache->get_id();
      model->_tensor_map[key_cache->get_id()] = std::move(key_cache);

      auto value_cache = std::make_unique<Tensor>(*reqs[b].value_cache[l]);
      value_cache->set_produced();
      model->_value_cache_tensor_ids[b][l] = value_cache->get_id();
      model->_tensor_map[value_cache->get_id()] = std::move(value_cache);
    }
  }
  return model;
}


void LanguageModel::register_operation(std::unique_ptr<Operation> op) {
  _operation_map[op->get_id()] = std::move(op);
}

std::unique_ptr<Tensor> LanguageModel::create_tensor(std::string name, std::vector<uint32_t> dims) {
  return std::make_unique<Tensor>(_root_node_id, name, dims, _config.precision, true);
}

std::unique_ptr<Tensor> LanguageModel::create_weight(std::string name, std::vector<uint32_t> dims) {
  auto tensor = std::make_unique<Tensor>(_root_node_id, name, dims, _config.precision, true);
  _wgt_map[name] = tensor->get_id();
  return std::move(tensor);
}

uint32_t LanguageModel::load_key_cache(uint32_t layer, uint32_t batch) {
  return _key_cache_tensor_ids[batch][layer];
}

uint32_t LanguageModel::load_value_cache(uint32_t layer, uint32_t batch) {
  return _value_cache_tensor_ids[batch][layer];
}

void LanguageModel::initialize_weight(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  for(int l = 0; l < _num_sim_layers; l++) {
    auto attn = name_gen(LAYER(l), BlockType::Attention);
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::LayerNorm, ParameterType::Weight), {_hidden_size})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::LayerNorm, ParameterType::Bias), {_hidden_size})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::QKVGen, ParameterType::Weight), {_hidden_size, _qkv_out_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::QKVGen, ParameterType::Bias), {_qkv_out_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::Projection, ParameterType::Weight), {_proj_in_dim, _hidden_size})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::Projection, ParameterType::Bias), {_hidden_size})));

    auto ffn = name_gen(LAYER(l), BlockType::FeedForward);
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::LayerNorm, ParameterType::Weight), {_hidden_size})));
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::LayerNorm, ParameterType::Bias), {_hidden_size})));

    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Weight), {_hidden_size, _ffn1_out_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Bias), {_ffn1_out_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Weight), {_intermediate_size, _hidden_size})));
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Bias), {_hidden_size})));
  }

  weight_table.push_back(std::move(create_weight(name_gen(OperationType::LmHead, ParameterType::Weight), {_hidden_size, _model_config["vocab_size"]})));

  _wgt_size = 0;
  for (auto& wgt : weight_table) {
    if(_run_single_layer && wgt->get_name() != name_gen(OperationType::LmHead, ParameterType::Weight)) {
      _wgt_size += ((uint64_t)wgt->get_size()) * _num_layers;
    }
    else {
      _wgt_size += wgt->get_size();
    }
  }
  _act_size = std::max(_qkv_out_dim, _ffn1_out_dim) * _config.precision;
  spdlog::info("Tensor Parallelsim {}, Pipeline Parallelism {}", _tensor_parallel_size, _pipeline_parallel_size);
  spdlog::info("Weight size: {:.2f} GB", (_wgt_size / (1.0 GB)));
}

void LanguageModel::initialize_model(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<uint32_t> input_lengthes;
  uint32_t num_tokens = 0;
  for(auto& req : _reqs) {
    num_tokens += req.seq_length;
    input_lengthes.push_back(req.seq_length);
  }
  std::vector<uint32_t> act_dim = {num_tokens, _hidden_size};
  std::map<std::string, std::string> qkv_attr  = {
    {"has_bias", "1"},
    {"input_shape", dims_to_string(act_dim)},
    {"weight_shape", dims_to_string({_hidden_size,_qkv_out_dim})},
    {"output_shape", dims_to_string({num_tokens, _qkv_out_dim})}};
  std::map<std::string, std::string> kv_concat_attr = {
    {"input_token_lengths", dims_to_string(input_lengthes)},
    {"num_kv_heads", std::to_string(_num_kv_heads)},
    {"num_heads", std::to_string(_num_heads)},
    {"hidden_size", std::to_string(_hidden_size)}};
  std::map<std::string, std::string> attention_attr = {
    {"num_tokens", std::to_string(0)}, // define for each batch
    {"num_heads", std::to_string(_num_heads)},
    {"num_kv_heads", std::to_string(_num_kv_heads)},
    {"hidden_size", std::to_string(_hidden_size/_tensor_parallel_size)}};
  std::map<std::string, std::string> concat_attr = {
    {"axis", std::to_string(0)}};
  std::map<std::string, std::string> proj_attr = {
    {"has_bias", "1"},
    {"input_shape", dims_to_string({num_tokens, _proj_in_dim})},
    {"weight_shape", dims_to_string({_proj_in_dim, _hidden_size})},
    {"output_shape", dims_to_string(act_dim)}};
  std::map<std::string, std::string> ffn1_attr = {
    {"has_bias", "0"},
    {"input_shape", dims_to_string({num_tokens, _hidden_size})},
    {"weight_shape", dims_to_string({_hidden_size, _ffn1_out_dim})},
    {"output_shape", dims_to_string({num_tokens, _ffn1_out_dim})}};
  std::map<std::string, std::string> bias_act_attr = {
    {"has_bias", "1"},
    {"activation", _model_config["activation_function"]},
    {"llama_mlp", std::to_string(_llama_mlp)}};

  std::map<std::string, std::string> ffn2_attr = {
    {"has_bias", "1"},
    {"input_shape", dims_to_string({num_tokens, _intermediate_size})},
    {"weight_shape", dims_to_string({_intermediate_size, _hidden_size})},
    {"output_shape", dims_to_string({num_tokens, _hidden_size})}};

  _input_tensor = create_tensor("input", act_dim);
  uint32_t input_id = _input_tensor->get_id();
  _tensor_map[_input_tensor->get_id()] = std::move(_input_tensor);
  for(auto it = weight_table.begin(); it != weight_table.end(); it++) {
    //initialize weights
    auto tensor = std::make_unique<Tensor>(*it->get());
    tensor->set_produced();
    uint32_t id = tensor->get_id();
    _tensor_map[id] = std::move(tensor);
  }

  std::map<std::string, std::string> empty_attr;
  for(int l = 0; l < _num_sim_layers; l++) {
    //QKV Proejction
    std::string qkv_name = name_gen(LAYER(l), BlockType::Attention, OperationType::QKVGen);
    uint32_t qkv_weight_id = _wgt_map[name_gen(qkv_name, ParameterType::Weight)];
    uint32_t qkv_bias_id = _wgt_map[name_gen(qkv_name, ParameterType::Bias)];
    auto qkv_op = std::make_unique<GemmWS>(_config, (Model*) this, qkv_name, qkv_attr, _target_core);
    qkv_op->add_input(input_id);
    qkv_op->add_input(qkv_weight_id);
    qkv_op->add_input(qkv_bias_id);
    qkv_op->initialize_tiles(_mapping_table);
    uint32_t qkv_output_id = qkv_op->get_output(0)->get_id();
    register_operation(std::move(qkv_op));
    //KV Cache
    auto kv_cache_op = std::make_unique<KVCacheConcat>(
      _config, (Model*) this, name_gen(LAYER(l), BlockType::Attention, OperationType::KVCacheConcat), kv_concat_attr, _target_core);
    kv_cache_op->add_input(qkv_output_id);
    for(int b = 0; b < _num_batch; b++) {
      uint32_t key_cache_id = load_key_cache(l, b);
      uint32_t value_cache_id = load_value_cache(l, b);
      kv_cache_op->add_input(key_cache_id);
      kv_cache_op->add_input(value_cache_id);
    }
    kv_cache_op->initialize_tiles(_mapping_table);
    std::vector<uint32_t> queries;
    std::vector<uint32_t> keys;
    std::vector<uint32_t> values;
    for(int b = 0; b < _num_batch; b++) {
      queries.push_back(kv_cache_op->get_output(b*3)->get_id());
      keys.push_back(kv_cache_op->get_output(b*3+1)->get_id());
      values.push_back(kv_cache_op->get_output(b*3+2)->get_id());
    }
    register_operation(std::move(kv_cache_op));
    //Attention
    std::vector<uint32_t> attention_outs;
    for(int b = 0; b < _num_batch; b++) {
      std::string attn_name = name_gen(LAYER(l), BlockType::Attention, OperationType::Attention, std::to_string(b));
      attention_attr["num_tokens"] = std::to_string(input_lengthes[b]);
      auto attn_op = std::make_unique<Attention>(_config, (Model*) this, attn_name, attention_attr, _target_core);
      attn_op->add_input(queries[b]);
      attn_op->add_input(keys[b]);
      attn_op->add_input(values[b]);
      attn_op->initialize_tiles(_mapping_table);
      uint32_t attn_output_id = attn_op->get_output(0)->get_id();
      attention_outs.push_back(attn_output_id);
      register_operation(std::move(attn_op));
    }
    //Concatenate attention outputs
    std::string attn_concat_name = name_gen(LAYER(l), BlockType::Attention, OperationType::AttentionConcat);
    auto attn_concat_op = std::make_unique<Concat>(_config, (Model*) this, attn_concat_name, concat_attr, _target_core);
    for(int b = 0; b < _num_batch; b++) {
      attn_concat_op->add_input(attention_outs[b]);
    }
    attn_concat_op->initialize_tiles(_mapping_table);
    uint32_t attn_concat_output_id = attn_concat_op->get_output(0)->get_id();
    register_operation(std::move(attn_concat_op));
    //Projection
    std::string proj_name = name_gen(LAYER(l), BlockType::Attention, OperationType::Projection);
    uint32_t proj_weight_id = _wgt_map[name_gen(proj_name, ParameterType::Weight)];
    uint32_t proj_bias_id = _wgt_map[name_gen(proj_name, ParameterType::Bias)];
    auto proj_op = std::make_unique<GemmWS>(_config, (Model*) this, proj_name, proj_attr, _target_core);
    proj_op->add_input(attn_concat_output_id);
    proj_op->add_input(proj_weight_id);
    proj_op->add_input(proj_bias_id);
    proj_op->initialize_tiles(_mapping_table);
    uint32_t proj_output_id = proj_op->get_output(0)->get_id();
    register_operation(std::move(proj_op));
    //Residual + LayerNorm
    std::string ln_name = name_gen(LAYER(l), BlockType::Attention, OperationType::LayerNorm);
    uint32_t ln_weight_id = _wgt_map[name_gen(ln_name, ParameterType::Weight)];
    uint32_t ln_bias_id = _wgt_map[name_gen(ln_name, ParameterType::Bias)];
    auto ln_op = std::make_unique<SkipLayerNorm>(_config, (Model*) this, ln_name, empty_attr, _target_core);
    ln_op->add_input(input_id);
    ln_op->add_input(proj_output_id);
    ln_op->add_input(ln_weight_id);
    ln_op->add_input(ln_bias_id);
    ln_op->initialize_tiles(_mapping_table);
    uint32_t ln_output_id = ln_op->get_output(0)->get_id();
    register_operation(std::move(ln_op));
    //FeedForward
    std::string ffn_name = name_gen(LAYER(l), BlockType::FeedForward);
    uint32_t ffn1_weight_id = _wgt_map[name_gen(ffn_name, OperationType::FullyConnected1, ParameterType::Weight)];
    uint32_t ffn1_bias_id = _wgt_map[name_gen(ffn_name, OperationType::FullyConnected1, ParameterType::Bias)];
    auto ffn1_op = std::make_unique<GemmWS>(
      _config, (Model*) this, name_gen(ffn_name, OperationType::FullyConnected1), ffn1_attr, _target_core);
    ffn1_op->add_input(ln_output_id);
    ffn1_op->add_input(ffn1_weight_id);
    ffn1_op->initialize_tiles(_mapping_table);
    uint32_t ffn1_output_id = ffn1_op->get_output(0)->get_id();
    register_operation(std::move(ffn1_op));
    //Gelu
    std::string act_name = name_gen(LAYER(l), BlockType::FeedForward, OperationType::Act);
    auto act_op = std::make_unique<BiasAct>(_config, (Model*) this, act_name, bias_act_attr, _target_core);
    act_op->add_input(ffn1_output_id);
    act_op->add_input(ffn1_bias_id);
    act_op->initialize_tiles(_mapping_table);
    uint32_t act_output_id = act_op->get_output(0)->get_id();
    register_operation(std::move(act_op));
    //FullyConnected2
    uint32_t ffn2_weight_id = _wgt_map[name_gen(ffn_name, OperationType::FullyConnected2, ParameterType::Weight)];
    uint32_t ffn2_bias_id = _wgt_map[name_gen(ffn_name, OperationType::FullyConnected2, ParameterType::Bias)];
    auto ffn2_op = std::make_unique<GemmWS>(
      _config, (Model*) this, name_gen(ffn_name, OperationType::FullyConnected2), ffn2_attr, _target_core);
    ffn2_op->add_input(act_output_id);
    ffn2_op->add_input(ffn2_weight_id);
    ffn2_op->add_input(ffn2_bias_id);
    ffn2_op->initialize_tiles(_mapping_table);
    uint32_t ffn2_output_id = ffn2_op->get_output(0)->get_id();
    register_operation(std::move(ffn2_op));
    //Residual + LayerNorm
    std::string ff_ln_name = name_gen(LAYER(l), BlockType::FeedForward, OperationType::LayerNorm);
    uint32_t ff_ln_weight_id = _wgt_map[name_gen(ff_ln_name, ParameterType::Weight)];
    uint32_t ff_ln_bias_id = _wgt_map[name_gen(ff_ln_name, ParameterType::Bias)];
    auto ff_ln_op = std::make_unique<SkipLayerNorm>(_config, (Model*) this, ff_ln_name, empty_attr, _target_core);
    ff_ln_op->add_input(ln_output_id);
    ff_ln_op->add_input(ffn2_output_id);
    ff_ln_op->add_input(ff_ln_weight_id);
    ff_ln_op->add_input(ff_ln_bias_id);
    ff_ln_op->initialize_tiles(_mapping_table);
    uint32_t ff_ln_output_id = ff_ln_op->get_output(0)->get_id();
    register_operation(std::move(ff_ln_op));
    input_id = ff_ln_output_id;
  }

  for (auto& [key, val]: _operation_map) {
    if(val->check_executable()) {
      spdlog::debug("runnable op, {}", val->get_optype());
      _executable_layer.push_back(val.get());
    }
  }
  /* Model initialization time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("{} Model initialization time: {:2f} seconds", _onnx_path, duration.count());
}

