#include "Interconnect.h"
#include "booksim2/Interconnect.hpp"
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

SimpleInterconnect::SimpleInterconnect(SimulationConfig config)
  :  _latency(config.icnt_latency) {
  spdlog::info("Initialize SimpleInterconnect");
  _cycles = 0;
  _config = config;
  _n_nodes = config.num_cores + config.dram_channels;
  _in_buffers.resize(_n_nodes);
  _out_buffers.resize(_n_nodes);
  _busy_node.resize(_n_nodes);
  for(int node = 0; node < _n_nodes; node++) {
    _busy_node[node] = false;
  }
}


bool SimpleInterconnect::running() {
  return false;
}

void SimpleInterconnect::cycle() {
  for(int node = 0; node < _n_nodes; node++) {
    int src_node = (_rr_start + node ) % _n_nodes;
    if(!_in_buffers[src_node].empty() && _in_buffers[src_node].front().finish_cycle <= _cycles) {
      uint32_t dest = _in_buffers[src_node].front().dest;
      if(!_busy_node[dest]) {
        _out_buffers[dest].push(_in_buffers[src_node].front().access);  
        _in_buffers[src_node].pop();
        _busy_node[dest] = true;
      }
    }
  }
  
  for(int node = 0; node < _n_nodes; node++) {
    _busy_node[node] = false;
  }
  _rr_start = (_rr_start + 1) % _n_nodes;
  _cycles++;
}

void SimpleInterconnect::push(uint32_t src, uint32_t dest, MemoryAccess* request) {
  SimpleInterconnect::Entity entity;
  if(_in_buffers[src].empty())
    entity.finish_cycle =  _cycles + _latency;
  else
    entity.finish_cycle =  _in_buffers[src].back().finish_cycle + 1;
  entity.dest = dest;
  entity.access = request;
  _in_buffers[src].push(entity);
}

bool SimpleInterconnect::is_full(uint32_t nid, MemoryAccess* request) {
  //TODO: limit buffersize
  return false;
}

bool SimpleInterconnect::is_empty(uint32_t nid) {
  return _out_buffers[nid].empty();
}

MemoryAccess* SimpleInterconnect::top(uint32_t nid) {
  assert(!is_empty(nid));
  return _out_buffers[nid].front();
}

void SimpleInterconnect::pop(uint32_t nid) {
  _out_buffers[nid].pop();
  // spdlog::trace("PUSH {}", _cycles);
}


Booksim2Interconnect::Booksim2Interconnect(SimulationConfig config) {
  _config = config;
  _n_nodes = config.num_cores + config.dram_channels;
  spdlog::info("Initialize Booksim2"); 
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
    std::string(onnxim_path_env) : std::string("./");

  _config_path = fs::path(onnxim_path).append("configs").append((std::string)config.icnt_config_path).string();
  spdlog::info("Config path : {}", _config_path);
  _booksim = std::make_unique<booksim2::Interconnect>(_config_path, _n_nodes);
  _ctrl_size = 8;
}

bool Booksim2Interconnect::running() {
  return false;
}

void Booksim2Interconnect::cycle() {
  _booksim->run();
}

void Booksim2Interconnect::push(uint32_t src, uint32_t dest, MemoryAccess* request) {
  booksim2::Interconnect::Type type = get_booksim_type(request);
  uint32_t size = get_packet_size(request);
  _booksim->push(request, 0, 0, size, type, src, dest);
}

bool Booksim2Interconnect::is_full(uint32_t nid, MemoryAccess* request) {
  uint32_t size = get_packet_size(request);
  return _booksim->is_full(nid, 0, size);
}

bool Booksim2Interconnect::is_empty(uint32_t nid) {
  return _booksim->is_empty(nid, 0);
}

MemoryAccess* Booksim2Interconnect::top(uint32_t nid) {
  assert(!is_empty(nid));
  return (MemoryAccess*) _booksim->top(nid, 0);
}

void Booksim2Interconnect::pop(uint32_t nid) {
  assert(!is_empty(nid));
  _booksim->pop(nid, 0);
}

void Booksim2Interconnect::print_stats() {
  _booksim->print_stats();
}

booksim2::Interconnect::Type Booksim2Interconnect::get_booksim_type(MemoryAccess* access) {
  booksim2::Interconnect::Type type;
  if(access->write && access->request) {
    /* Write request */
    type = booksim2::Interconnect::Type::WRITE;
  }
  else if(access->write && !access->request) {
    /* Write response */
    type = booksim2::Interconnect::Type::WRITE_REPLY;
  }
  else if(!access->write && access->request){
    /* Read request */
    type = booksim2::Interconnect::Type::READ;
  } 
  else if(!access->write && !access->request) {
    /* Read reply */
    type = booksim2::Interconnect::Type::READ_REPLY;
  }
  return type;
}

uint32_t Booksim2Interconnect::get_packet_size(MemoryAccess* access) {
  uint32_t size;
  if(access->write && access->request) {
    /* Write request */
    size = access->size;
  }
  else if(access->write && !access->request) {
    /* Write response */
    size = _ctrl_size;
  }
  else if(!access->write && access->request){
    /* Read request */
    size = _ctrl_size;
  } 
  else if(!access->write && !access->request) {
    /* Read reply */
    size = access->size;
  }
  return size;
}