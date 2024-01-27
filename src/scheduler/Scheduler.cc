#include "Scheduler.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle)
    : _config(config), _core_cycle(core_cycle) {}

void Scheduler::schedule_model(std::unique_ptr<Model> model,
                               uint32_t sample_size) {
  _request_queue.push_back(Request{.request_id = generate_id(),
                                   .model = std::move(model),
                                   .sample_size = sample_size});
  spdlog::info("MODEL {} Scheduled, Total Request: {}",
               _request_queue.back().model->get_name(), _request_queue.size());
  refresh_status();
}

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
    _request_queue.front().model->set_layer_finish(tile.layer_id);
    _layer_stat_map[tile.layer_id] = _active_layers_map[tile.layer_id];
    _active_layers_map.erase(tile.layer_id);
  }
  refresh_status();
}

bool Scheduler::empty() { return _request_queue.empty(); }

void Scheduler::refresh_status() {
  if (!_request_queue.empty()) {
    if (_request_queue.front().model->check_finish()) {
      spdlog::info("Model finish  at {}", *_core_cycle);
      _request_queue.pop_front();
    }
  }
  if (!_request_queue.empty() && _executable_tile_queue.empty() &&
      count_active_layers() == 0) {
    spdlog::info("executable layer count {}",
                 _request_queue.front().model->get_executable_layers().size());
    Operation* new_layer =
        _request_queue.front().model->get_executable_layers().front();
    spdlog::info("Start layer {}", new_layer->get_name().c_str());
    for (int output_id = 0; output_id < new_layer->num_outputs(); output_id++)
      new_layer->get_output(output_id)->allocate_tensor(_config.precision);
    assert(new_layer->get_tiles().size());
    _executable_tile_queue = new_layer->get_tiles();
    _active_layers_map[new_layer->get_id()] =
        LayerStat{.id = new_layer->get_id(),
                  .name = new_layer->get_name(),
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
    std::string model_name;
    bool model_finish = false;
    for (int req_index = 0; req_index < _request_queue.size(); req_index++) {
      if (_request_queue[req_index].request_id ==
          _active_layers_map[tile.layer_id].request_id) {
        model_finish = true;
        _request_queue[req_index].model->set_layer_finish(tile.layer_id);
        model_name = _request_queue[req_index].model->get_name();
      }
    }
    spdlog::info("Layer {} {} finish at {}", model_name,
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
  if (!_request_queue.empty()) {
    for (auto req = _request_queue.begin(); req != _request_queue.end();
         req++) {
      if (req->model->check_finish()) {
        spdlog::info("Model[{}] finish  at {}", req->model->get_name(),
                     *_core_cycle);
        req = _request_queue.erase(req);
        --req;
      }
    }
  }
  if (!_request_queue.empty() && _executable_tile_queue.empty()) {
    _request_rr = _request_rr % _request_queue.size();
    Operation* new_layer =
        _request_queue[_request_rr].model->get_executable_layers().front();
    if (_active_layers_map.find(new_layer->get_id()) ==
        _active_layers_map.end()) {
      if (count_active_layers() > 0)
        spdlog::info("Layer {} {}: launched before finish prior layer",
                     new_layer->get_name(), new_layer->get_id());
      else
        spdlog::info("Layer {} {}: Enqueue", new_layer->get_name(),
                     new_layer->get_id());
      for (int output_id = 0; output_id < new_layer->num_outputs(); output_id++)
        new_layer->get_output(output_id)->allocate_tensor(_config.precision);
      // new_layer->initialize_tiles(_config);
      assert(new_layer->get_tiles().size());
      _executable_tile_queue = new_layer->get_tiles();
      _active_layers_map[new_layer->get_id()] = LayerStat{
          .id = new_layer->get_id(),
          .request_id = _request_queue[_request_rr].request_id,
          .name = new_layer->get_name(),
          .launched = false,
          .start_cycle = *_core_cycle,
          .remain_tiles = (uint32_t)_executable_tile_queue.size(),
      };
      _request_rr = (_request_rr + 1) % _request_queue.size();
      return;
    }
  }
}

HalfSplitScheduler::HalfSplitScheduler(SimulationConfig config,
                                       const cycle_type* core_cycle)
    : Scheduler(config, core_cycle) {}

void HalfSplitScheduler::schedule_model(std::unique_ptr<Model> model,
                                        uint32_t sample_size) {
  _request_queue.push_back(Request{.request_id = generate_id(),
                                   .model = std::move(model),
                                   .sample_size = sample_size});
  spdlog::info("MODEL {} Scheduled, Total Request: {}",
               _request_queue.back().model->get_name(), _request_queue.size());
  _executable_tile_queue_table[_request_queue.back().request_id] =
      std::deque<Tile>();
  refresh_status();
}

Tile HalfSplitScheduler::get_tile(uint32_t core_id) {
  uint32_t target_id = core_id % _request_queue.size();
  uint32_t req_id = _request_queue[target_id].request_id;
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
    std::string model_name;
    bool model_finish = false;
    for (int req_index = 0; req_index < _request_queue.size(); req_index++) {
      if (_request_queue[req_index].request_id ==
          _active_layers_map[tile.layer_id].request_id) {
        model_finish = true;
        _request_queue[req_index].model->set_layer_finish(tile.layer_id);
        model_name = _request_queue[req_index].model->get_name();
        _executable_tile_queue_table.erase(
            _request_queue[req_index].request_id);
      }
    }
    spdlog::info("Layer {} {} finish at {}", model_name,
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
  if (!_request_queue.empty()) {
    for (auto req = _request_queue.begin(); req != _request_queue.end();
         req++) {
      if (req->model->check_finish()) {
        spdlog::info("Model[{}] finish  at {}", req->model->get_name(),
                     *_core_cycle);
        req = _request_queue.erase(req);
        --req;
      }
    }
  }
  if (!_request_queue.empty()) {
    for (auto req = _request_queue.begin(); req != _request_queue.end();
         req++) {
      if (_executable_tile_queue_table[req->request_id].empty()) {
        Operation* new_layer = req->model->get_executable_layers().front();
        if (_active_layers_map.find(new_layer->get_id()) ==
            _active_layers_map.end()) {
          if (count_active_layers() > 0)
            spdlog::info("Layer {} {}: launched before finish prior layer",
                         new_layer->get_name(), new_layer->get_id());
          else
            spdlog::info("Layer {} {}: Enqueue", new_layer->get_name(),
                         new_layer->get_id());
          for (int output_id = 0; output_id < new_layer->num_outputs();
               output_id++)
            new_layer->get_output(output_id)->allocate_tensor(
                _config.precision);
          // new_layer->initialize_tiles(_config);
          assert(new_layer->get_tiles().size());
          _executable_tile_queue_table[req->request_id] =
              new_layer->get_tiles();
          _active_layers_map[new_layer->get_id()] = LayerStat{
              .id = new_layer->get_id(),
              .request_id = req->request_id,
              .name = new_layer->get_name(),
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
}
