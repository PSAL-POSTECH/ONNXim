#ifndef LANGUAGE_SCHEDULER_H
#define LANGUAGE_SCHEDULER_H
#include <string>

#include "../Common.h"
#include "../models/LanguageModel.h"

struct LangRequest {
  uint32_t request_id;
  bool running;
  bool gen_phase;
  uint64_t request_time;
  uint64_t start_time;
  uint64_t finish_time;
  uint32_t prompt_length;
  uint32_t current_length;
  uint32_t target_length;
  std::vector<std::unique_ptr<Tensor>> key_cache;
  std::vector<std::unique_ptr<Tensor>> value_cache;
};

class LangScheduler {
  public:
    LangScheduler(std::string name, std::string path, std::unique_ptr<LanguageModel> model, SimulationConfig config);
    bool can_schedule_model();
    virtual std::unique_ptr<Model> pop_model();
    virtual void finish_model(uint32_t model_id);
    virtual void cycle();
    virtual bool busy();
  protected:
    SimulationConfig _config;
    std::string _name;
    std::unique_ptr<LanguageModel> _language_model;
    std::queue<std::unique_ptr<LangRequest>> _request_queue;
    std::map<uint32_t, std::unique_ptr<LangRequest>> _active_requests;
    std::map<uint32_t, std::vector<uint32_t>> _requests_in_model;
    std::queue<std::unique_ptr<Model>> _model_queue;
    uint64_t _cycle;

    uint32_t _num_layers;
    uint32_t _num_attention_heads;
    uint32_t _num_kv_heads;
    uint32_t _hidden_size;
    uint32_t _cache_dim;
    uint32_t _max_seq_length;

    void parse_request_trace(std::string trace_path);
    
};

#endif