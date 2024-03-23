#include "Sram.h"
#define NUM_PORTS 3

Sram::Sram(SimulationConfig config, const cycle_type& core_cycle, bool accum)
    : _core_cycle(core_cycle) {
  if (!accum) {
    _size = config.spad_size;
  } else {
    _size = config.accum_spad_size;
  }
  _data_width = config.dram_req_size;
  int precision = config.precision;
  _current_size[0] = 0; // multiple of _data_width
  _current_size[1] = 0;
  _accum = accum;
}

bool Sram::check_hit(addr_type address, int buffer_id) {
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end())
    return false;
  _cache_table[buffer_id][address].timestamp = _core_cycle;
  return _cache_table[buffer_id][address].valid;
}

bool Sram::check_full(int buffer_id) {
  return _current_size[buffer_id] < _size KB / _data_width / 2;
}

bool Sram::check_remain(size_t size, int buffer_id) {
  return _current_size[buffer_id] + size <= _size KB / _data_width / 2;
}

bool Sram::check_allocated(addr_type address, int buffer_id) {
  return _cache_table[buffer_id].find(address) != _cache_table[buffer_id].end();
}

void Sram::cycle() {}

void Sram::flush(int buffer_id) {
  _current_size[buffer_id] = 0;
  _cache_table[buffer_id].clear();
}

int Sram::prefetch(addr_type address, int buffer_id, size_t allocated_size,
                    size_t count) {
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end()) {
    if (!check_remain(allocated_size, buffer_id)) {
      print_all(buffer_id);
      assert(0);
      return 0;
    }
    _current_size[buffer_id] += allocated_size;
  } else if (_cache_table[buffer_id].find(address) !=
                 _cache_table[buffer_id].end() &&
             _accum) {
    assert(_cache_table[buffer_id][address].size == allocated_size);
    return 0;
  } else {
    assert(0);
    return 0;
  }

  _cache_table[buffer_id][address] = SramEntry{.valid = false,
                                               .size = allocated_size,
                                               .remain_req_count = count,
                                               .timestamp = _core_cycle};
  return 1;
}

void Sram::fill(addr_type address, int buffer_id) {
  assert(check_allocated(address, buffer_id));
  assert(_cache_table[buffer_id][address].remain_req_count > 0 &&
         !_cache_table[buffer_id][address].valid);
  _cache_table[buffer_id][address].remain_req_count--;
  if (_cache_table[buffer_id][address].remain_req_count == 0) {
    _cache_table[buffer_id][address].valid = true;
    spdlog::trace("MAKE valid {} {}F", buffer_id, address);
  }
}

void Sram::count_up(addr_type address, int buffer_id) {
  assert(check_allocated(address, buffer_id));
  _cache_table[buffer_id][address].remain_req_count++;
  if (_cache_table[buffer_id][address].valid) {
    _cache_table[buffer_id][address].valid = false;
    spdlog::trace("MAKE valid {} {}F", buffer_id, address);
  }
}

void Sram::print_all(int buffer_id) {
  for (auto& [key, val] : _cache_table[buffer_id]) {
    spdlog::info("{:x} : {}", key, val.size);
  }
}