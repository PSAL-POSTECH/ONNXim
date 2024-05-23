#ifndef KV_CACHE_CONCAT_H
#define KV_CACHE_CONCAT_H
#include "Operation.h"

class KVCacheConcat : public Operation {
  public:
    KVCacheConcat(SimulationConfig config, Model* model,
                  onnx::NodeProto& node_proto);
    KVCacheConcat(const KVCacheConcat& src);
    KVCacheConcat(SimulationConfig config, Model* model, std::string name,
                  std::map<std::string, std::string>& attributes);
    void initialize_tiles(MappingTable& mapping_table) override;
  private:
  uint32_t _num_batches;
  std::vector<uint32_t> _input_token_lengths;
  uint32_t _num_kv_heads;
  uint32_t _num_attention_heads;
  uint32_t _hidden_size;
  uint32_t _cache_dim;
};

#endif