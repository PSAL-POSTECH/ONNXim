#ifndef LanguageModel_H
#define LanguageModel_H

#include "../Common.h"
#include "../Model.h"

#define LAYER(i) ("layer" + std::to_string(i))

namespace BlockType {
extern std::string Attention;
extern std::string FeedForward;
}  // namespace BlockType

namespace OperationType {
extern std::string LayerNorm;
extern std::string QKVGen;
extern std::string Projection;
extern std::string FullyConnected1;
extern std::string FullyConnected2;
extern std::string LmHead;
extern std::string Act;
extern std::string KVCacheConcat;
extern std::string AttentionConcat;
extern std::string Attention;
}  // namespace OperationType

namespace ParameterType {
extern std::string Weight;
extern std::string Bias;
}  // namespace ParameterType


struct LanguageRequest {
  uint32_t seq_length;
  uint32_t context_length;
  std::vector<Tensor*> key_cache;
  std::vector<Tensor*> value_cache;
};

class LanguageModel : public Model {
  public:
  LanguageModel(json llm_config, SimulationConfig config, std::string name);
  std::unique_ptr<LanguageModel> generate_model(std::vector<LanguageRequest> &reqs);
  
  uint32_t load_key_cache(uint32_t layer, uint32_t batch);
  uint32_t load_value_cache(uint32_t layer, uint32_t batch);
  
  void log_model();
  uint64_t get_weight_size() { return _wgt_size; }

  virtual void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table);
  virtual void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table);
  protected:
    std::vector<LanguageRequest> _reqs;
    uint32_t _num_batch;
    bool _generation_phase;

    uint32_t _num_heads;
    uint32_t _num_kv_heads;
    uint32_t _hidden_size;
    uint32_t _intermediate_size;
    uint32_t _num_layers;

    uint64_t _wgt_size;  // in bytes

    std::unique_ptr<Tensor> _input_tensor;

    robin_hood::unordered_map<std::string, uint32_t> _wgt_map;
    std::vector<std::vector<uint32_t>> _key_cache_tensor_ids;
    std::vector<std::vector<uint32_t>> _value_cache_tensor_ids;
   
    void register_operation(std::unique_ptr<Operation>);
    std::unique_ptr<Tensor> create_tensor(std::string name,
                                        std::vector<uint32_t> dims);
    std::unique_ptr<Tensor> create_weight(std::string name,
                                        std::vector<uint32_t> dims);
};

#endif