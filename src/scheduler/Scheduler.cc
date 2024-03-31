#include "Scheduler.h"

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time)
    : _config(config), _core_cycle(core_cycle), _core_time(core_time) {
  _core_executable_tile_queue.resize(_config.num_cores);
  _partition_map = config.partiton_map;
}

void Scheduler::schedule_model(std::unique_ptr<Model> model,
                               uint32_t sample_size) {
  _request_queue.push_back(Request{.request_id = generate_id(),
                                   .model = std::move(model),
                                   .sample_size = sample_size});
  spdlog::info("MODEL {} Scheduled, Total Request: {}",
               _request_queue.back().model->get_name(), _request_queue.size());
  refresh_status();
}

void Scheduler::issue_tile_per_core(std::vector<uint32_t>& allowed_cpu, int offset) {
  while(!_executable_tile_queue.empty()) {
    Tile tile = _executable_tile_queue.front();
    /* Barrier! */
    if (tile.status == Tile::Status::BAR)
      break;

    uint32_t core_id = offset;
    if (tile.core_id == -1) { // -1 is global id
      core_id += _core_rr_id;
      _core_rr_id = _core_rr_id + 1; // increase with round robin
    } else {
      core_id = core_id + _nr_layer;
    }
    core_id = allowed_cpu[core_id % allowed_cpu.size()];
    tile.core_id = core_id;
    _core_executable_tile_queue[core_id].push_back(tile);
    _executable_tile_queue.pop_front();
  }
}

void Scheduler::issue_tile_per_core() {
  while(!_executable_tile_queue.empty()) {
    Tile tile = _executable_tile_queue.front();
    /* Barrier! */
    if (tile.status == Tile::Status::BAR)
      break;

    if (tile.core_id == -1) { // -1 is global id
      tile.core_id = _core_rr_id;
      _core_rr_id = (_core_rr_id + 1) % _config.num_cores; // increase with round robin
    } else {
      tile.core_id = (tile.core_id + _nr_layer) % _config.num_cores;
    }
    _core_executable_tile_queue[tile.core_id].push_back(tile);
    _executable_tile_queue.pop_front();
  }
}

/*TODO: Add base address for each addr in tiles */
Tile Scheduler::get_tile(uint32_t core_id) {
  if (_core_executable_tile_queue[core_id].empty() && _executable_tile_queue.empty()) {
    Tile tile;
    tile.status = Tile::Status::EMPTY;
    return tile;
  } else {
    if (!_core_executable_tile_queue[core_id].empty()) {
      Tile tile = _core_executable_tile_queue[core_id].front();
      _active_layers_map[tile.layer_id].launched_tiles++;
      _core_executable_tile_queue[core_id].pop_front();
      spdlog::debug("Layer {} Core {} Get Tile at {}", _active_layers_map[tile.layer_id].name, core_id,
                    *_core_cycle);
      return tile;
    } else {
      Tile tile = _executable_tile_queue.front();
      if (tile.status == Tile::Status::BAR) {
        LayerStat stat = _active_layers_map[tile.layer_id];
        if (stat.launched_tiles == stat.finished_tiles) {
          /* POP only if all lauched tiles are finished */
          _executable_tile_queue.pop_front();
          _active_layers_map[tile.layer_id].launched_tiles++;
          _active_layers_map[tile.layer_id].finished_tiles++;
          _active_layers_map[tile.layer_id].remain_tiles--;
          if (!_active_layers_map[tile.layer_id].remain_tiles) {
            _active_layers_map[tile.layer_id].remain_tiles++;
            finish_tile(core_id, tile);
          } else {
            /* Issue to core scheduler blocked by barrier */
            uint32_t request_id = _active_layers_map[_executable_tile_queue.front().layer_id].request_id;
            uint32_t partition_id = 0;
            uint32_t offset = 0;
            for (int req_index = 0; req_index < _request_queue.size(); req_index++) {
              if (_request_queue[req_index].request_id == request_id) {
                partition_id = _request_queue[req_index].model->get_partition_id();
                break;
              }
            }
            std::vector<uint32_t>& allowed_cpu = _partition_map.at(partition_id);
            auto it = find(allowed_cpu.begin(), allowed_cpu.end(), core_id);
            offset = it - allowed_cpu.begin();
            issue_tile_per_core(allowed_cpu, offset);
          }
        }
        Tile empty_tile{.status = Tile::Status::EMPTY};
        return empty_tile;
      } else {
        spdlog::error("[Scheudler] Something wrong happned...!");
        Tile empty_tile{.status = Tile::Status::EMPTY};
        return empty_tile;
      }
    }
  }
}

bool Scheduler::is_accum_tile(uint32_t core_id, int index) {
  if (_core_executable_tile_queue[core_id].size() <= index) {
    return false;
  } else {
    return _core_executable_tile_queue[core_id].at(index).accum;
  }
}

bool Scheduler::tile_queue_empty() {
  bool all_empty = true;
  for (int i = 0; i < _config.num_cores; i++) {
    all_empty = all_empty && _core_executable_tile_queue[i].empty();
  }
  all_empty = all_empty && _executable_tile_queue.empty();
  return all_empty;
}

void Scheduler::finish_tile(uint32_t core_id, Tile tile) {
  spdlog::debug("Layer {} Core {} Finish Tile at {} Remain tile {}", tile.layer_id, core_id,
                *_core_cycle, _active_layers_map[tile.layer_id].remain_tiles);
  assert(_active_layers_map.find(tile.layer_id) != _active_layers_map.end());
  assert(_active_layers_map[tile.layer_id].remain_tiles > 0);
  _active_layers_map[tile.layer_id].remain_tiles--;
  _active_layers_map[tile.layer_id].finished_tiles++;

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
      spdlog::info("Model[{}] Request: {} us, Start: {} us, finish:{} us, Current Cycle:{}",
                    _request_queue.front().model->get_name(),
                    _request_queue.front().model->get_request_time() / 1000000,
                    _request_queue.front().model->get_start_time() / 1000000,
                    (*_core_time) / 1000000, *_core_cycle);
      _request_queue.pop_front();
    }
  }
  bool all_empty = tile_queue_empty();
  if (!_request_queue.empty() && all_empty &&
      count_active_layers() == 0) {
    spdlog::info("executable layer count {}",
                 _request_queue.front().model->executable_layer_size());
    Operation* new_layer =
        _request_queue.front().model->get_executable_tile();

    spdlog::info("Start layer {}", new_layer->get_name().c_str());
    for (int output_id = 0; output_id < new_layer->num_outputs(); output_id++) {
      new_layer->get_output(output_id)->set_produced();
    }
    _request_queue.front().model->update_start_time(*_core_time);
    /* Get tiles from new layer */
    assert(new_layer->get_tiles().size());
    _executable_tile_queue = new_layer->get_tiles();
    _nr_layer++;
    _active_layers_map[new_layer->get_id()] =
        LayerStat{.id = new_layer->get_id(),
                  .name = new_layer->get_name(),
                  .launched = true,
                  .start_cycle = *_core_cycle,
                  .total_tiles = (uint32_t)_executable_tile_queue.size(),
                  .remain_tiles = (uint32_t)_executable_tile_queue.size(),
                  .finished_tiles = 0,
                  .launched_tiles = 0};

    /* Issue tiles to core scheduler */
    issue_tile_per_core();
  }
}

uint32_t Scheduler::count_active_layers() {
  uint32_t count = 0;
  count = _active_layers_map.size();
  return count;
}

DedicatedCPUScheduler::DedicatedCPUScheduler(SimulationConfig config,
                                       const cycle_type* core_cycle, const uint64_t* core_time)
    : TimeMultiplexScheduler(config, core_cycle, core_time) {}

void DedicatedCPUScheduler::refresh_status() {
  if (!_request_queue.empty()) {
    for (auto req = _request_queue.begin(); req != _request_queue.end();
         req++) {
      if (req->model->check_finish()) {
        spdlog::info("Model[{}] Request: {} us, Start: {} us, finish:{} us Current Cycle:{}",
                      req->model->get_name(),
                      req->model->get_request_time()/1000000,
                      req->model->get_start_time()/1000000,
                      (*_core_time)/1000000, *_core_cycle);
        req = _request_queue.erase(req);
        --req;
      }
    }
  }

  if (!_request_queue.empty()) {
    for (int i=0; i<_config.num_cores; i++) {
      /* If core queue is empty, issue a new layer */
      if (!_core_executable_tile_queue[i].empty())
        continue;

      Operation* new_layer = nullptr;
      uint32_t partition_id = 0;
      uint32_t offset = 0;
      std::vector<uint32_t> allowed_cpu;
      int req_index;
      for (req_index=0; req_index<_request_queue.size(); req_index++) {
        partition_id = _request_queue[req_index].model->get_partition_id();
        allowed_cpu = _partition_map.at(partition_id);
        auto it = find(allowed_cpu.begin(), allowed_cpu.end(), i);
        if (it != allowed_cpu.end()) {
          new_layer = _request_queue[req_index].model->get_executable_tile();
          /* Set cpu offset */
          offset = it - allowed_cpu.begin();
          break;
        }
      }

      /* Check layer is found */
      if (new_layer == nullptr)
        continue;

      if (_active_layers_map.find(new_layer->get_id()) ==
          _active_layers_map.end()) {
        if (count_active_layers() > 0)
          spdlog::info("Layer {} {}: launched before finish prior layer",
                      new_layer->get_name(), new_layer->get_id());
        else
          spdlog::info("Layer {} {}: Enqueue", new_layer->get_name(),
                      new_layer->get_id());

        for (int output_id = 0; output_id < new_layer->num_outputs(); output_id++) {
          new_layer->get_output(output_id)->set_produced();
        }
        _request_queue[req_index].model->update_start_time(*_core_time);
        // new_layer->initialize_tiles(_config);
        assert(new_layer->get_tiles().size());
        _executable_tile_queue = new_layer->get_tiles();
        _nr_layer++;
        _active_layers_map[new_layer->get_id()] =
            LayerStat{.id = new_layer->get_id(),
                      .request_id = _request_queue[req_index].request_id,
                      .name = new_layer->get_name(),
                      .launched = true,
                      .start_cycle = *_core_cycle,
                      .total_tiles = (uint32_t)_executable_tile_queue.size(),
                      .remain_tiles = (uint32_t)_executable_tile_queue.size(),
                      .finished_tiles = 0,
                      .launched_tiles = 0};
        issue_tile_per_core(allowed_cpu, offset);
      }
    }
  }
}

TimeMultiplexScheduler::TimeMultiplexScheduler(SimulationConfig config,
                                       const cycle_type* core_cycle, const uint64_t* core_time)
    : Scheduler(config, core_cycle, core_time) {}

void TimeMultiplexScheduler::finish_tile(uint32_t core_id, Tile tile) {
  assert(_active_layers_map.find(tile.layer_id) != _active_layers_map.end());
  assert(_active_layers_map[tile.layer_id].remain_tiles > 0);
  _active_layers_map[tile.layer_id].remain_tiles--;
  _active_layers_map[tile.layer_id].finished_tiles++;

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
        spdlog::info("Model[{}] Request: {} us, Start: {} us, finish:{} us Current Cycle:{}",
                      req->model->get_name(),
                      req->model->get_request_time()/1000000,
                      req->model->get_start_time()/1000000,
                      (*_core_time)/1000000, *_core_cycle);
        req = _request_queue.erase(req);
        --req;
      }
    }
  }
  bool all_empty = tile_queue_empty();
  if (!_request_queue.empty() && all_empty) {
    _request_rr = (_request_rr + 1) % _request_queue.size();
    Operation* new_layer =
        _request_queue[_request_rr].model->get_executable_tile();
    if (_active_layers_map.find(new_layer->get_id()) ==
        _active_layers_map.end()) {
      if (count_active_layers() > 0)
        spdlog::info("Layer {} {}: launched before finish prior layer",
                     new_layer->get_name(), new_layer->get_id());
      else
        spdlog::info("Layer {} {}: Enqueue", new_layer->get_name(),
                     new_layer->get_id());

      for (int output_id = 0; output_id < new_layer->num_outputs(); output_id++) {
        new_layer->get_output(output_id)->set_produced();
      }
      _request_queue[_request_rr].model->update_start_time(*_core_time);
      // new_layer->initialize_tiles(_config);
      assert(new_layer->get_tiles().size());
      _executable_tile_queue = new_layer->get_tiles();
      _nr_layer++;
      _active_layers_map[new_layer->get_id()] =
          LayerStat{.id = new_layer->get_id(),
                    .request_id = _request_queue[_request_rr].request_id,
                    .name = new_layer->get_name(),
                    .launched = true,
                    .start_cycle = *_core_cycle,
                    .total_tiles = (uint32_t)_executable_tile_queue.size(),
                    .remain_tiles = (uint32_t)_executable_tile_queue.size(),
                    .finished_tiles = 0,
                    .launched_tiles = 0};

      /* Issue tiles to core scheduler */
      issue_tile_per_core();
    }
  }
}

HalfSplitScheduler::HalfSplitScheduler(SimulationConfig config,
                                       const cycle_type* core_cycle, const uint64_t* core_time)
    : Scheduler(config, core_cycle, core_time) {}

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
  assert(_active_layers_map[tile.layer_id].remain_tiles > 0);
  _active_layers_map[tile.layer_id].remain_tiles--;
  _active_layers_map[tile.layer_id].finished_tiles++;

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
        Operation* new_layer = req->model->get_executable_tile();
        if (_active_layers_map.find(new_layer->get_id()) ==
            _active_layers_map.end()) {
          if (count_active_layers() > 0)
            spdlog::info("Layer {} {}: launched before finish prior layer",
                         new_layer->get_name(), new_layer->get_id());
          else
            spdlog::info("Layer {} {}: Enqueue", new_layer->get_name(),
                         new_layer->get_id());
          for (int output_id = 0; output_id < new_layer->num_outputs();
               output_id++) {
            new_layer->get_output(output_id)->set_produced();
          }
          // new_layer->initialize_tiles(_config);
          assert(new_layer->get_tiles().size());
          _executable_tile_queue_table[req->request_id] =
              new_layer->get_tiles();
          _active_layers_map[new_layer->get_id()] =
              LayerStat{.id = new_layer->get_id(),
                        .request_id = req->request_id,
                        .name = new_layer->get_name(),
                        .launched = false,
                        .start_cycle = *_core_cycle,
                        .total_tiles = (uint32_t)_executable_tile_queue_table[req->request_id].size(),
                        .remain_tiles = (uint32_t)_executable_tile_queue_table[req->request_id].size(),
                        .finished_tiles = 0,
                        .launched_tiles = 0};
        }
      }
    }
  }
}
