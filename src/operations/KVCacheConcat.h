#ifndef KV_CACHE_CONCAT_H
#define KV_CACHE_CONCAT_H
#include "Operation.h"

class KVCacheConcat : public Operation {
  public:
    KVCacheConcat(SimulationConfig config, Model* model,
                  onnx::NodeProto& node_proto, uint32_t target_core=0);
    KVCacheConcat(const KVCacheConcat& src);
    KVCacheConcat(SimulationConfig config, Model* model, std::string name,
                  std::map<std::string, std::string>& attributes, uint32_t target_core=0);
    void initialize_tiles(MappingTable& mapping_table) override;
  private:
    void calculate_loops();
    void initialize_instructions(Tile* tile, uint32_t idx);

    uint32_t _num_batches;
    std::vector<uint32_t> _input_token_lengths;
    uint32_t _num_kv_heads;
    uint32_t _num_attention_heads;
    uint32_t _hidden_size;
    uint32_t _cache_dim;
    uint32_t _outter_loops;
    uint32_t _inner_loops;
};

#endif