#include "Scheduler.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle)
    : _config(config), _core_cycle(core_cycle) {}

void Scheduler::schedule_tile(std::unique_ptr<TileGraph> tile_graph, uint32_t sample_size) {
  _tile_request_queue.push_back(Tile_Request{.request_id = generate_id(),
                                   .tile_graph = std::move(tile_graph),
                                   .sample_size = sample_size});
  spdlog::info("Tile scheduled, Total Request: {}", _tile_request_queue.size());
  refresh_status();
}

/*TODO: Add base address for each addr in tiles */
Tile Scheduler::get_tile(uint32_t core_id) {
  if (_executable_tile_queue.empty()) {
    Tile tile;
    tile.status = Tile::Status::EMPTY;
    return tile;
  } else {
    Tile tile = _executable_tile_queue.front();
    if (tile.status == Tile::Status::BAR) {
      LayerStat stat = _layer_stat_map[tile.layer_id];
      if (stat.launched_tiles + stat.remain_tiles == stat.total_tiles) {
        /* POP only if all lauched tiles are finished */
        _executable_tile_queue.pop_front();
        _layer_stat_map[tile.layer_id].launched_tiles++;
        _layer_stat_map[tile.layer_id].remain_tiles--;
      }
      Tile empty_tile{.status = Tile::Status::EMPTY};
      return empty_tile;
    } else {
      _layer_stat_map[tile.layer_id].launched_tiles++;
      _executable_tile_queue.pop_front();
      spdlog::debug("Layer {} Core {} Get Tile at {}", tile.layer_id, core_id,
                    *_core_cycle);
      return tile;
    }
  }
}

void Scheduler::finish_tile(uint32_t core_id, Tile tile) {
  spdlog::debug("Layer {} Core {} Finish Tile at {}", tile.layer_id, core_id,
                *_core_cycle);
  assert(_active_layers_map.find(tile.layer_id) != _active_layers_map.end());
  assert(_layer_stat_map.find(tile.layer_id) == _layer_stat_map.end());
  assert(_active_layers_map[tile.layer_id].remain_tiles > 0);
  _active_layers_map[tile.layer_id].remain_tiles--;

  if (_active_layers_map[tile.layer_id].remain_tiles == 0) {
    _active_layers_map[tile.layer_id].finish_cycle = *_core_cycle;
    spdlog::info("Layer {} finish at {}",
                 _active_layers_map[tile.layer_id].name, *_core_cycle);
    spdlog::info("Total compute time {}",
                 *_core_cycle - _active_layers_map[tile.layer_id].start_cycle);
    _tile_request_queue.front().tile_graph->set_finish();
    _layer_stat_map[tile.layer_id] = _active_layers_map[tile.layer_id];
    _active_layers_map.erase(tile.layer_id);
  }
  refresh_status();
}

bool Scheduler::empty() { return _tile_request_queue.empty(); }

void Scheduler::refresh_status() {
  if (!_tile_request_queue.empty()) {
    if (_tile_request_queue.front().tile_graph->check_finish()) {
      spdlog::info("Kernel finish  at {}", *_core_cycle);
      _tile_request_queue.pop_front();
    }
  }
  if (!_tile_request_queue.empty() && _executable_tile_queue.empty() &&
      count_active_layers() == 0) {
    _executable_tile_queue = _tile_request_queue.front().tile_graph->get_tiles();
    _active_layers_map[_tile_request_queue.front().tile_graph->get_id()] =
        LayerStat{.id = _tile_request_queue.front().tile_graph->get_id(),
                  .name = "element-wise",
                  .launched = true,
                  .start_cycle = *_core_cycle,
                  .total_tiles = (uint32_t)_executable_tile_queue.size(),
                  .remain_tiles = (uint32_t)_executable_tile_queue.size(),
                  .launched_tiles = 0};
  }
}

uint32_t Scheduler::count_active_layers() {
  uint32_t count = 0;
  count = _active_layers_map.size();
  return count;
}

TimeMultiplexScheduler::TimeMultiplexScheduler(SimulationConfig config,
                                               const cycle_type* core_cycle)
    : Scheduler(config, core_cycle) {
  _request_rr = 0;
}

Tile TimeMultiplexScheduler::get_tile(uint32_t core_id) {
  if (_executable_tile_queue.empty()) {
    Tile tile;
    tile.status = Tile::Status::EMPTY;
    return tile;
  } else {
    Tile tile = _executable_tile_queue.front();
    _executable_tile_queue.pop_front();
    if (!_active_layers_map[tile.layer_id].launched) {
      _active_layers_map[tile.layer_id].launched = true;
      _active_layers_map[tile.layer_id].start_cycle = *_core_cycle;
      spdlog::info("Start layer {}", _active_layers_map[tile.layer_id].name);
    }
    return tile;
  }
}

void TimeMultiplexScheduler::finish_tile(uint32_t core_id, Tile tile) {
  assert(_active_layers_map.find(tile.layer_id) != _active_layers_map.end());
  assert(_layer_stat_map.find(tile.layer_id) == _layer_stat_map.end());
  assert(_active_layers_map[tile.layer_id].remain_tiles > 0);
  _active_layers_map[tile.layer_id].remain_tiles--;

  if (_active_layers_map[tile.layer_id].remain_tiles == 0) {
    _active_layers_map[tile.layer_id].finish_cycle = *_core_cycle;
    std::string kernel_name = "kernel0"; // TODO: kernel name
    bool model_finish = false;
    for (int req_index = 0; req_index < _tile_request_queue.size(); req_index++) {
      if (_tile_request_queue[req_index].request_id ==
          _active_layers_map[tile.layer_id].request_id) {
        model_finish = true;
        _tile_request_queue[req_index].tile_graph->set_finish();
      }
    }
    spdlog::info("Layer {} {} finish at {}", kernel_name,
                 _active_layers_map[tile.layer_id].name, *_core_cycle);
    spdlog::info("Total compute time {}",
                 *_core_cycle - _active_layers_map[tile.layer_id].start_cycle);
    assert(model_finish);
    _layer_stat_map[tile.layer_id] = _active_layers_map[tile.layer_id];
    _active_layers_map.erase(tile.layer_id);
  }
  refresh_status();
}

void TimeMultiplexScheduler::refresh_status() {
  std::string kernel_name = "kernel0";
  if (!_tile_request_queue.empty()) {
    for (auto req = _tile_request_queue.begin(); req != _tile_request_queue.end();
         req++) {
      if (req->tile_graph->check_finish()) {
        spdlog::info("Model[{}] finish  at {}", kernel_name,
                     *_core_cycle);
        req = _tile_request_queue.erase(req);
        --req;
      }
    }
  }
  if (!_tile_request_queue.empty() && _executable_tile_queue.empty()) {
    _request_rr = _request_rr % _tile_request_queue.size();
    _executable_tile_queue = _tile_request_queue[_request_rr].tile_graph->get_tiles();
    _active_layers_map[_tile_request_queue.front().tile_graph->get_id()] = LayerStat{
        .id = _tile_request_queue.front().tile_graph->get_id(),
        .request_id = _tile_request_queue[_request_rr].request_id,
        .name = kernel_name,
        .launched = false,
        .start_cycle = *_core_cycle,
        .remain_tiles = (uint32_t)_executable_tile_queue.size(),
    };
    _request_rr = (_request_rr + 1) % _tile_request_queue.size();
    return;
  }
}

HalfSplitScheduler::HalfSplitScheduler(SimulationConfig config,
                                       const cycle_type* core_cycle)
    : Scheduler(config, core_cycle) {}

Tile HalfSplitScheduler::get_tile(uint32_t core_id) {
  uint32_t target_id = core_id % _tile_request_queue.size();
  uint32_t req_id = _tile_request_queue[target_id].request_id;
  if (_executable_tile_queue_table[req_id].empty()) {
    Tile tile;
    tile.status = Tile::Status::EMPTY;
    return tile;
  } else {
    Tile tile = _executable_tile_queue_table[req_id].front();
    _executable_tile_queue_table[req_id].pop_front();
    if (!_active_layers_map[tile.layer_id].launched) {
      _active_layers_map[tile.layer_id].launched = true;
      _active_layers_map[tile.layer_id].start_cycle = *_core_cycle;
      spdlog::info("Start layer {}", _active_layers_map[tile.layer_id].name);
    }
    return tile;
  }
}

void HalfSplitScheduler::finish_tile(uint32_t core_id, Tile tile) {
  assert(_active_layers_map.find(tile.layer_id) != _active_layers_map.end());
  assert(_layer_stat_map.find(tile.layer_id) == _layer_stat_map.end());
  assert(_active_layers_map[tile.layer_id].remain_tiles > 0);
  _active_layers_map[tile.layer_id].remain_tiles--;

  if (_active_layers_map[tile.layer_id].remain_tiles == 0) {
    _active_layers_map[tile.layer_id].finish_cycle = *_core_cycle;
    std::string kernel_name = "kernel0"; // TODO: kernel name
    bool model_finish = false;
    for (int req_index = 0; req_index < _tile_request_queue.size(); req_index++) {
      if (_tile_request_queue[req_index].request_id ==
          _active_layers_map[tile.layer_id].request_id) {
        model_finish = true;
        _tile_request_queue[req_index].tile_graph->set_finish();
        _executable_tile_queue_table.erase(
            _tile_request_queue[req_index].request_id);
      }
    }
    spdlog::info("Layer {} {} finish at {}", kernel_name,
                 _active_layers_map[tile.layer_id].name, *_core_cycle);
    spdlog::info("Total compute time {}",
                 *_core_cycle - _active_layers_map[tile.layer_id].start_cycle);
    assert(model_finish);
    _layer_stat_map[tile.layer_id] = _active_layers_map[tile.layer_id];
    _active_layers_map.erase(tile.layer_id);
  }
  refresh_status();
}

void HalfSplitScheduler::refresh_status() {
  std::string kernel_name = "kernel0";
  if (!_tile_request_queue.empty()) {
    for (auto req = _tile_request_queue.begin(); req != _tile_request_queue.end();
         req++) {
      if (req->tile_graph->check_finish()) {
        spdlog::info("Model[{}] finish  at {}", kernel_name,
                     *_core_cycle);
        req = _tile_request_queue.erase(req);
        --req;
      }
    }
  }
  if (!_tile_request_queue.empty()) {
    for (auto req = _tile_request_queue.begin(); req != _tile_request_queue.end();
         req++) {
      if (_executable_tile_queue_table[req->request_id].empty()) {
        _executable_tile_queue_table[req->request_id] =
            _tile_request_queue.front().tile_graph->get_tiles();
        _active_layers_map[_tile_request_queue.front().tile_graph->get_id()] = LayerStat{
            .id = _tile_request_queue.front().tile_graph->get_id(),
            .request_id = req->request_id,
            .name = kernel_name,
            .launched = false,
            .start_cycle = *_core_cycle,
            .remain_tiles =
                (uint32_t)_executable_tile_queue_table[req->request_id]
                    .size(),
        };
      }
    }
  }
}
