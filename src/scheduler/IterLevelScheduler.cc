#include "IterLevelScheduler.h"

IterLevelScheduler::IterLevelScheduler(std::string name, std::string path, 
                             std::unique_ptr<LanguageModel> model,
                             SimulationConfig config,
                             json scheduler_config) 
  : LangScheduler(name, path, std::move(model), config, scheduler_config) {
}

void IterLevelScheduler::cycle() {
  _cycle++;
  if(_active_requests.size() <= _max_batch_size || _max_batch_size == 0) {
    while(!_request_queue.empty()) {
      if(_request_queue.front()->request_time <= _cycle) {
        init_request(_request_queue.front());
        _active_requests[_request_queue.front()->request_id] = std::move(_request_queue.front());
        _request_queue.pop();
      }
      else {
        break;
      }
      if(_max_batch_size > 0 && _active_requests.size() >= _max_batch_size) {
        break;
      }
    }
  }

  if(_model_queue.empty() && _requests_in_model.empty()) {
    init_inputs_and_model();
  }
}

