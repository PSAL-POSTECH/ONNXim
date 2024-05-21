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
  _llm_config = llm_config;
  _num_batch = 0;
  _num_token = 0;
  _target_token = 0;
  _is_decode = false;
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

uint32_t LanguageModel::load_cache(uint32_t layer, uint32_t batch) {
  std::string cache_name = name_gen(LAYER(layer), BlockType::Attention, OperationType::KVCacheConcat);
  // return _wgt_map[cache_name];
  return 0;
}

void LanguageModel::initialize_weight(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  int layers = _model_config["num_hidden_layers"];
  int emb_dim = _model_config["hidden_size"];
  int ffn_dim = _model_config["intermediate_size"];
  bool llama_mlp = _model_config["ffn_type"] == "llama";
  for(int l = 0; l < layers; l++) {
    auto attn = name_gen(LAYER(l), BlockType::Attention);
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::LayerNorm, ParameterType::Weight), {emb_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::LayerNorm, ParameterType::Bias), {emb_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::QKVGen, ParameterType::Weight), {emb_dim, 3*emb_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::QKVGen, ParameterType::Bias), {3*emb_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::Projection, ParameterType::Weight), {emb_dim, emb_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(attn, OperationType::Projection, ParameterType::Bias), {emb_dim})));

    auto ffn = name_gen(LAYER(l), BlockType::FeedForward);
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::LayerNorm, ParameterType::Weight), {emb_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::LayerNorm, ParameterType::Bias), {emb_dim})));
    int ffn1_out_dim = ffn_dim;
    if(llama_mlp) {
      ffn1_out_dim *= 2;
    }
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Weight), {emb_dim, ffn1_out_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::FullyConnected1, ParameterType::Bias), {ffn1_out_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Weight), {ffn_dim, emb_dim})));
    weight_table.push_back(std::move(create_weight(name_gen(ffn, OperationType::FullyConnected2, ParameterType::Bias), {emb_dim})));
  }

  weight_table.push_back(std::move(create_weight(name_gen(OperationType::LmHead, ParameterType::Weight), {emb_dim, _model_config["vocab_size"]})));
  
  _wgt_size = 0;
  for (auto& wgt : weight_table) {
    _wgt_size += wgt->get_size();
  }

  spdlog::info("Weight size: {}", _wgt_size);
}

void LanguageModel::initialize_model(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  //TODO: make it modulear
  _input_tensor = create_tensor("input", {1, _model_config["max_seq_length"], _model_config["hidden_size"]});//TODO:: fix this
  int input_id = _input_tensor->get_id();
  std::map<std::string, std::string> empty_attr;
  for(int l = 0; l < _num_layers; l++) {
    //QKV Proejction
    std::string qkv_name = name_gen(LAYER(l), BlockType::Attention, OperationType::QKVGen);
    uint32_t qkv_weight_id = _wgt_map[name_gen(qkv_name, ParameterType::Weight)];
    uint32_t qkv_bias_id = _wgt_map[name_gen(qkv_name, ParameterType::Bias)];
    auto qkv_op = std::make_unique<GemmWS>(_config, (Model*) this, qkv_name, empty_attr);
    qkv_op->add_input(input_id);
    qkv_op->add_input(qkv_weight_id);
    qkv_op->add_input(qkv_bias_id);
    qkv_op->initialize_tiles(_mapping_table);
    uint32_t qkv_output_id = qkv_op->get_output(0)->get_id();
    register_operation(std::move(qkv_op));
    //KV Cache
    auto kv_cache_op = std::make_unique<KVCacheConcat>(
      _config, (Model*) this, name_gen(LAYER(l), BlockType::Attention, OperationType::KVCacheConcat), empty_attr);
    kv_cache_op->add_input(qkv_output_id);
    for(int b = 0; b < _num_batch; b++) {
      uint32_t kv_cache_id = load_cache(l, b);
      kv_cache_op->add_input(kv_cache_id);
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
      std::string attn_name = name_gen(LAYER(l), BlockType::Attention, OperationType::Attention);
      auto attn_op = std::make_unique<Attention>(_config, (Model*) this, attn_name, empty_attr); //TODO: add attribute
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
    auto attn_concat_op = std::make_unique<Concat>(_config, (Model*) this, attn_concat_name, empty_attr);
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
    auto proj_op = std::make_unique<GemmWS>(_config, (Model*) this, proj_name, empty_attr);
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
    auto ln_op = std::make_unique<SkipLayerNorm>(_config, (Model*) this, ln_name, empty_attr);
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
      _config, (Model*) this, name_gen(ffn_name, OperationType::FullyConnected1), empty_attr);
    ffn1_op->add_input(ln_output_id);
    ffn1_op->add_input(ffn1_weight_id);
    ffn1_op->initialize_tiles(_mapping_table);
    uint32_t ffn1_output_id = ffn1_op->get_output(0)->get_id();
    register_operation(std::move(ffn1_op));
    //Gelu
    std::string act_name = name_gen(LAYER(l), BlockType::FeedForward, OperationType::Act);
    auto act_op = std::make_unique<BiasAct>(_config, (Model*) this, act_name, empty_attr);
    act_op->add_input(ffn1_output_id);
    act_op->add_input(ffn1_bias_id);
    act_op->initialize_tiles(_mapping_table);
    uint32_t act_output_id = act_op->get_output(0)->get_id();
    register_operation(std::move(act_op));
    //FullyConnected2
    uint32_t ffn2_weight_id = _wgt_map[name_gen(ffn_name, OperationType::FullyConnected2, ParameterType::Weight)];
    uint32_t ffn2_bias_id = _wgt_map[name_gen(ffn_name, OperationType::FullyConnected2, ParameterType::Bias)];
    auto ffn2_op = std::make_unique<GemmWS>(
      _config, (Model*) this, name_gen(ffn_name, OperationType::FullyConnected2), empty_attr);
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
    auto ff_ln_op = std::make_unique<SkipLayerNorm>(_config, (Model*) this, ff_ln_name, empty_attr);
    ff_ln_op->add_input(ln_output_id);
    ff_ln_op->add_input(ffn2_output_id);
    ff_ln_op->add_input(ff_ln_weight_id);
    ff_ln_op->add_input(ff_ln_bias_id);
    ff_ln_op->initialize_tiles(_mapping_table);
    uint32_t ff_ln_output_id = ff_ln_op->get_output(0)->get_id();
    register_operation(std::move(ff_ln_op));
    input_id = ff_ln_output_id;
  }
  //LMHead
  uint32_t lm_head_weight_id = _wgt_map[name_gen(OperationType::LmHead, ParameterType::Weight)];
  auto lm_head_op = std::make_unique<GemmWS>(_config, (Model*) this, OperationType::LmHead, empty_attr);
  lm_head_op->add_input(input_id);
  lm_head_op->add_input(lm_head_weight_id);
  lm_head_op->initialize_tiles(_mapping_table);
  uint32_t lm_head_output_id = lm_head_op->get_output(0)->get_id();
  register_operation(std::move(lm_head_op));
}

