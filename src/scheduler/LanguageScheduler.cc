#include "LanguageScheduler.h"
#include <fstream>

LangScheduler::LangScheduler(std::string name, std::string path, std::unique_ptr<LanguageModel> model) {
  _name = name;
  _language_model = std::move(model);
  _cycle = 0;
  parse_request_trace(path); 
}

bool LangScheduler::can_schedule_model() {
  return !_model_queue.empty();
}

std::unique_ptr<Model> LangScheduler::pop_model() {
  std::unique_ptr<Model> model = std::move(_model_queue.front());
  _model_queue.pop();
  return model;
}

void LangScheduler::cycle() {
  _cycle++;
  //Reqeust Queue to Active Requests if active is empty
  if(!_request_queue.empty()) {
    if(_request_queue.front()->request_time <= _cycle && _active_requests.empty()) {
      _request_queue.front()->start_time = _cycle;
      _active_requests[_request_queue.front()->request_id] = std::move(_request_queue.front());
      _request_queue.pop();
    }
  }

  //Active Requests to Model Queue
  if(_model_queue.empty()) {
    std::vector<LangInput> inputs;
    for(auto it = _active_requests.begin(); it != _active_requests.end();) {
      if(it->second->running == false) {
        LangInput input;
        input.request_id = it->first;
        if(it->second->gen_phase) {
          input.seq_length = 1;
          input.context_length = it->second->current_length;
        }
        else {
          input.seq_length = it->second->prompt_length;
          input.context_length = 0;
        }
        inputs.push_back(input);
      }
    }
    auto infer_model = _language_model->generate_model(inputs);
    for(auto input : inputs) {
      _active_requests[input.request_id]->running = true;
      _requests_in_model[infer_model->get_id()].push_back(input.request_id);
    }
    _model_queue.push(std::move(infer_model));
  }
}

void LangScheduler::finish_model(uint32_t model_id) {
  for(auto req_id : _requests_in_model[model_id]) {
    if(!_active_requests[req_id]->gen_phase) {
      _active_requests[req_id]->gen_phase = true;
      _active_requests[req_id]->current_length = _active_requests[req_id]->prompt_length;
    }
    else {
      _active_requests[req_id]->current_length += 1;
    }
    _active_requests[req_id]->running = false;
    if(_active_requests[req_id]->current_length == _active_requests[req_id]->target_length) {
      _active_requests[req_id]->finish_time = _cycle;
      spdlog::info("Request {} completed in {} cycles", req_id, _active_requests[req_id]->finish_time - _active_requests[req_id]->start_time);
      _active_requests.erase(req_id);
    }
  }
  _requests_in_model.erase(model_id);
}

bool LangScheduler::busy() {
  return !_model_queue.empty() || !_active_requests.empty() || !_request_queue.empty();
}

void LangScheduler::parse_request_trace(std::string path) {
  std::ifstream trace_file(path);
  if (!trace_file.is_open()) {
    spdlog::error("Failed to open trace file: {}", path);
    return;
  }
  //Parse CSV input (prompt_length, target_length) and create LangRequest objects
  std::string line;
  uint32_t id = 0;
  std::getline(trace_file, line); //Skip header
  while (std::getline(trace_file, line)) {
    std::stringstream ss(line);
    std::string prompt_length, target_length;
    std::getline(ss, prompt_length, ',');
    std::getline(ss, target_length, ',');
    std::unique_ptr<LangRequest> request = std::make_unique<LangRequest>();
    request->request_id = id++;
    request->request_time = 0;
    request->start_time = 0;
    request->running = false;
    request->prompt_length = std::stoi(prompt_length);
    request->target_length = std::stoi(target_length);
    _request_queue.push(std::move(request));
  }
  trace_file.close();
}