#include "Dram.h"

#include "helper/HelperFunctions.h"
#include "Hashing.h"

uint32_t Dram::get_channel_id(MemoryAccess* access) {
  uint32_t channel_id;
  if (_n_ch >= 16)
    channel_id = ipoly_hash_function((new_addr_type)access->dram_address/_config.dram_req_size, 0, _n_ch);
  else
    channel_id = ipoly_hash_function((new_addr_type)access->dram_address/_config.dram_req_size, 0, 16) % _n_ch;
  return channel_id;
}

/* FIXME: Simple DRAM has bugs */
SimpleDram::SimpleDram(SimulationConfig config)
    : _latency(config.dram_latency) {
  _cycles = 0;
  _config = config;
  _n_ch = config.dram_channels;
  _waiting_queue.resize(_n_ch);
  _response_queue.resize(_n_ch);
}

bool SimpleDram::running() { return false; }

void SimpleDram::cycle() {
  for (uint32_t ch = 0; ch < _n_ch; ch++) {
    if (!_waiting_queue[ch].empty() &&
        _waiting_queue[ch].front().first <= _cycles) {
      _response_queue[ch].push(_waiting_queue[ch].front().second);
      _waiting_queue[ch].pop();
    }
  }

  _cycles++;
}

bool SimpleDram::is_full(uint32_t cid, MemoryAccess* request) { return false; }

void SimpleDram::push(uint32_t cid, MemoryAccess* request) {
  request->request = false;
  std::pair<uint64_t, MemoryAccess*> entity;
  entity.first = MAX(_cycles + _latency, _last_finish_cycle);
  _last_finish_cycle = entity.first;
  entity.second = request;
  _waiting_queue[cid].push(entity);
}

bool SimpleDram::is_empty(uint32_t cid) { return _response_queue[cid].empty(); }

MemoryAccess* SimpleDram::top(uint32_t cid) {
  assert(!is_empty(cid));
  return _response_queue[cid].front();
}

void SimpleDram::pop(uint32_t cid) {
  assert(!is_empty(cid));
  _response_queue[cid].pop();
}

DramRamulator::DramRamulator(SimulationConfig config)
    : _mem(std::make_unique<ram::Ramulator>(config.dram_config_path,
                                            config.num_cores, false)) {
  _n_ch = config.dram_channels;
  _config = config;
  _cycles = 0;
  _total_processed_requests.resize(_n_ch);
  _processed_requests.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _total_processed_requests[ch] = 0;
    _processed_requests[ch] = 0;
  }
}

bool DramRamulator::running() { return false; }

void DramRamulator::cycle() {
  _mem->tick();
  _cycles++;
  int interval = _config.dram_print_interval? _config.dram_print_interval: INT32_MAX;
  int average = 0;
  if (_cycles % interval == 0) {
    for (int ch = 0; ch < _n_ch; ch++) {
      float util = ((float)_processed_requests[ch]) / interval * 100;
      _total_processed_requests[ch] += _processed_requests[ch];
      average += _processed_requests[ch];
      _processed_requests[ch] = 0;
    }
    spdlog::info("Avg DRAM: BW Util {:.2f}%", (float)average / (interval * _n_ch) * 100);
  }
}

bool DramRamulator::is_full(uint32_t cid, MemoryAccess* request) {
  return !_mem->isAvailable(cid, request->dram_address, request->write);
}

void DramRamulator::push(uint32_t cid, MemoryAccess* request) {
  const addr_type atomic_bytes = _mem->getAtomicBytes();
  const addr_type target_addr = request->dram_address;
  // align address
  const addr_type start_addr = target_addr - (target_addr % atomic_bytes);
  assert(start_addr == target_addr);
  assert(request->size == atomic_bytes);
  int count = 0;
  request->request = false;
  _mem->push(cid, target_addr, request->write, request->core_id, request);
}

bool DramRamulator::is_empty(uint32_t cid) { return _mem->isEmpty(cid); }

MemoryAccess* DramRamulator::top(uint32_t cid) {
  assert(!is_empty(cid));
  return (MemoryAccess*)_mem->top(cid);
}

void DramRamulator::pop(uint32_t cid) {
  assert(!is_empty(cid));
  _mem->pop(cid);
  _processed_requests[cid]++;
}

void DramRamulator::print_stat() {
  uint32_t total_reqs = 0;
  for (int ch = 0; ch < _n_ch; ch++) {
    _total_processed_requests[ch] += _processed_requests[ch];
    float util = ((float)_total_processed_requests[ch]) / _cycles * 100;
    spdlog::info("DRAM CH[{}]: AVG BW Util {:.2f}%", ch, util);
    total_reqs += _total_processed_requests[ch];
  }
  float util = ((float)total_reqs / _n_ch) / _cycles * 100;
  spdlog::info("DRAM: AVG BW Util {:.2f}%", util);
  _mem->print_stats();
}

DramRamulator2::DramRamulator2(SimulationConfig config) {
  _n_ch = config.dram_channels;
  _req_size = config.dram_req_size;
  _config = config;
  _mem.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch] = std::make_unique<NDPSim::Ramulator2>(
      ch, _n_ch, config.dram_config_path, "Ramulator2", _config.dram_print_interval, 1);
  }
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch) + _tx_log2;
}

bool DramRamulator2::running() { 
  return false;
}

void DramRamulator2::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();
  }
}

bool DramRamulator2::is_full(uint32_t cid, MemoryAccess* request) {
  return _mem[cid]->full();
}

void DramRamulator2::push(uint32_t cid, MemoryAccess* request) {
  addr_type atomic_bytes =_config.dram_req_size;
  addr_type target_addr = request->dram_address;
  // align address
  addr_type start_addr = target_addr - (target_addr % atomic_bytes);
  assert(start_addr == target_addr);
  assert(request->size == atomic_bytes);
  target_addr = (target_addr >> _tx_ch_log2) << _tx_log2;
  NDPSim::mem_fetch* mf = new NDPSim::mem_fetch();
  mf->addr = target_addr;
  mf->size = request->size;
  mf->write = request->write;
  mf->request = true;
  mf->origin_data = request;
  _mem[cid]->push(mf);
}

bool DramRamulator2::is_empty(uint32_t cid) { 
  return _mem[cid]->return_queue_top() == NULL;
}

MemoryAccess* DramRamulator2::top(uint32_t cid) {
  assert(!is_empty(cid));
  NDPSim::mem_fetch* mf = _mem[cid]->return_queue_top();
  ((MemoryAccess*)mf->origin_data)->request = false;
  return (MemoryAccess*)mf->origin_data;
}

void DramRamulator2::pop(uint32_t cid) {
  assert(!is_empty(cid));
  NDPSim::mem_fetch* mf = _mem[cid]->return_queue_pop();
  delete mf;
}

void DramRamulator2::print_stat() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->print(stdout);
  }
}
