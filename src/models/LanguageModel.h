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
extern std::string QKVSplit;
extern std::string QKMatMul;
extern std::string SoftMax;
extern std::string LsVMatMul;
extern std::string AReshape;
extern std::string Residual;
extern std::string Gelu;
extern std::string BatchSplit;
extern std::string BatchConcat;

extern std::string KCacheConcat;
extern std::string VCacheConcat;
extern std::string VConcat;

extern std::string PIMGEMVSoftmax;
extern std::string PIMGEMVAdd;
extern std::string Microbench;
extern std::string NeuPIMSLogitSoftmax;
extern std::string Attention;
extern std::string NeuPIMSAttend;
extern std::string FusedMHA;
extern std::string PIMGEMV;
}  // namespace OperationType

namespace ParameterType {
extern std::string Weight;
extern std::string Bias;
}  // namespace ParameterType
class BatchedRequest;

class LanguageModel : public Model {
  public:
  LanguageModel(json llm_config, SimulationConfig config, std::string name);
  std::unique_ptr<LanguageModel> generate_model(BatchedRequest req);
  
  uint32_t load_cache(uint32_t layer, uint32_t batch);
  
  void log_model();
  uint64_t get_weight_size() { return _wgt_size; }

  virtual void initialize_model(
      std::vector<std::unique_ptr<Tensor>>& weight_table) = 0;
  virtual void initialize_weight(
      std::vector<std::unique_ptr<Tensor>>& weight_table) = 0;
  protected:
    json _llm_config;
    uint32_t _num_batch;
    uint32_t _num_token;
    uint32_t _target_token;
    bool _is_decode;

    uint32_t _num_heads;
    uint32_t _num_kv_heads;
    uint32_t _hidden_size;
    uint32_t _intermediate_size;
    uint32_t _num_layers;

    uint64_t _wgt_size;  // in bytes

    std::unique_ptr<Tensor> _input_tensor;

    robin_hood::unordered_map<std::string, uint32_t> _wgt_map;
   
    void register_operation(std::unique_ptr<Operation>);
    std::unique_ptr<Tensor> create_tensor(std::string name,
                                        std::vector<uint32_t> dims);
    std::unique_ptr<Tensor> create_weight(std::string name,
                                        std::vector<uint32_t> dims);
};

class OPTModel : public LanguageModel {
  public:
    OPTModel(json llm_config, SimulationConfig config, std::string name);
    OPTModel(BatchedRequest &request);
    virtual void initialize_weight(std::vector<std::unique_ptr<Tensor>>& weight_table) override;
    virtual void initialize_model(std::vector<std::unique_ptr<Tensor>>& weight_table) override;
  protected:

};

// class Llama2Model : public LanguageModel {
  //TODO:
//  public:
  // Llama2Model(json llm_config, SimulationConfig config, std::string name);
// };

#endif