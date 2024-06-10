#include "Scheduler.h"
#include "../Simulator.h"

std::unique_ptr<Scheduler> Scheduler::create(SimulationConfig config,
                                             const cycle_type* core_cycle, const uint64_t* core_time, void* simulator) {
  if (config.scheduler_type == "simple") {
    return std::make_unique<Scheduler>(config, core_cycle, core_time, simulator);
  } else if (config.scheduler_type == "partition_cpu") {
    return 
        std::make_unique<DedicatedCPUScheduler>(config, core_cycle, core_time, simulator);
  } else if (config.scheduler_type == "time_multiplex") {
    return 
        std::make_unique<TimeMultiplexScheduler>(config, core_cycle, core_time, simulator);
  } else if (config.scheduler_type == "spatial_split") {
    return std::make_unique<HalfSplitScheduler>(config, core_cycle, core_time, simulator);
  } else {
    spdlog::error("[Configuration] {} is invalid scheduler type...!", config.scheduler_type);
    exit(EXIT_FAILURE);
  }
}

Scheduler::Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time, void* simulator)
    : _config(config), _core_cycle(core_cycle), _core_time(core_time) {
  //_core_executable_tile_queue.resize(_config.num_cores);
  _partition_map = config.partiton_map;
  _simulator = simulator;
  //_executable_tile_queue.resize(_partition_map.size());
  for (const auto& pair: _partition_map) {
    uint32_t partition_id = pair.first;
    const std::vector<uint32_t> cpu_list = pair.second;
    for (const auto& cpu: cpu_list)
      _cpu_to_partition[cpu] = partition_id;
    _executable_tile_queue[partition_id] = std::deque<std::unique_ptr<Tile>>();
  }

  for (int i=0; i<config.num_cores;i++)
    _core_executable_tile_queue[i] = std::deque<std::unique_ptr<Tile>>();
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

uint32_t Scheduler::cpu_to_partition(uint32_t cpu) {
  return _cpu_to_partition[cpu];
}

void Scheduler::issue_tile_per_core(std::vector<uint32_t>& allowed_cpu, int offset, uint32_t partition_id) {
  while(!_executable_tile_queue[partition_id].empty()) {
    std::unique_ptr<Tile>& tile = _executable_tile_queue[partition_id].front();
    /* Barrier! */
    if (tile->status == Tile::Status::BAR)
      break;

    uint32_t core_id = offset;
    if (tile->core_id == -1) { // -1 is global id
      core_id += _core_rr_id;
      _core_rr_id = _core_rr_id + 1; // increase with round robin
    } else {
      core_id = tile->core_id + _nr_layer;
    }
    core_id = allowed_cpu[core_id % allowed_cpu.size()];
    spdlog::info("pushed to queue[{}], tile->core_id: {}", core_id, tile->core_id);
    tile->core_id = core_id;
    _core_executable_tile_queue[core_id].push_back(std::move(tile));
    _executable_tile_queue[partition_id].pop_front();
  }
}

void Scheduler::issue_tile_per_core() {
  while(!_executable_tile_queue[0].empty()) {
    std::unique_ptr<Tile>& tile = _executable_tile_queue[0].front();
    /* Barrier! */
    if (tile->status == Tile::Status::BAR)
      break;

    if (tile->core_id == -1) { // -1 is global id
      tile->core_id = _core_rr_id % _config.num_cores;
      _core_rr_id++; // increase with round robin
    } else {
      tile->core_id = (tile->core_id + _nr_layer) % _config.num_cores;
    }
    _core_executable_tile_queue[tile->core_id].push_back(std::move(tile));
    _executable_tile_queue[0].pop_front();
  }
}

/*TODO: Add base address for each addr in tiles */
std::unique_ptr<Tile> Scheduler::get_tile(uint32_t core_id) {
  uint32_t partition_id = cpu_to_partition(core_id);
  if (_core_executable_tile_queue[core_id].empty() && _executable_tile_queue[partition_id].empty()) {
    refresh_status();

    std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{});
    tile->status = Tile::Status::EMPTY;
    return tile;
  } else {
    if (!_core_executable_tile_queue[core_id].empty()) {
      std::unique_ptr<Tile> tile = std::move(_core_executable_tile_queue[core_id].front());
      _active_layers_map[tile->layer_id].launched_tiles++;
      _core_executable_tile_queue[core_id].pop_front();
      spdlog::debug("Layer {} Core {} Get Tile at {}", _active_layers_map[tile->layer_id].name, core_id,
                    *_core_cycle);
      return tile;
    } else {
      std::unique_ptr<Tile>& tile = _executable_tile_queue[partition_id].front();
      int layer_id = tile->layer_id;
      if (tile->status == Tile::Status::BAR) {
        LayerStat stat = _active_layers_map[layer_id];
        if (stat.launched_tiles == stat.finished_tiles) {
          /* POP only if all lauched tiles are finished */
          _executable_tile_queue[partition_id].pop_front();
          _active_layers_map[layer_id].launched_tiles++;
          _active_layers_map[layer_id].finished_tiles++;
          _active_layers_map[layer_id].remain_tiles--;
          if (!_active_layers_map[layer_id].remain_tiles) {
            _active_layers_map[layer_id].remain_tiles++;
            finish_tile(core_id, layer_id);
          } else {
            /* Issue to core scheduler blocked by barrier */
            uint32_t request_id = _active_layers_map[layer_id].request_id;
            uint32_t offset = 0;

            std::vector<uint32_t>& allowed_cpu = _partition_map.at(partition_id);
            auto it = find(allowed_cpu.begin(), allowed_cpu.end(), core_id);
            offset = it - allowed_cpu.begin();
            issue_tile_per_core(allowed_cpu, offset, partition_id);
          }
        }
        std::unique_ptr<Tile> empty_tile = std::make_unique<Tile>(Tile{.status = Tile::Status::EMPTY});
        return empty_tile;
      } else {
        spdlog::error("[Scheduler] Something wrong happened...! {}", tile->optype);
        std::unique_ptr<Tile> empty_tile = std::make_unique<Tile>(Tile{.status = Tile::Status::EMPTY});
        return empty_tile;
      }
    }
  }
}

bool Scheduler::is_accum_tile(uint32_t core_id, int index) {
  if (_core_executable_tile_queue[core_id].size() <= index) {
    return false;
  } else {
    return _core_executable_tile_queue[core_id].at(index)->accum;
  }
}

bool Scheduler::tile_queue_empty() {
  bool all_empty = true;
  for (int i = 0; i < _config.num_cores; i++) {
    all_empty = all_empty && _core_executable_tile_queue[i].empty();
  }
  all_empty = all_empty && _executable_tile_queue[0].empty();
  return all_empty;
}

void Scheduler::finish_tile(uint32_t core_id, int layer_id) {
  spdlog::debug("Layer {} Core {} Finish Tile at {} Remain tile {}", layer_id, core_id,
                *_core_cycle, _active_layers_map[layer_id].remain_tiles);
  assert(_active_layers_map.find(layer_id) != _active_layers_map.end());
  assert(_active_layers_map[layer_id].remain_tiles > 0);
  _active_layers_map[layer_id].remain_tiles--;
  _active_layers_map[layer_id].finished_tiles++;

  if (_active_layers_map[layer_id].remain_tiles == 0) {
    _active_layers_map[layer_id].finish_cycle = *_core_cycle;
    spdlog::info("Layer {} finish at {}",
                 _active_layers_map[layer_id].name, *_core_cycle);
    spdlog::info("Total compute time {}",
                 *_core_cycle - _active_layers_map[layer_id].start_cycle);
    _request_queue.front().model->set_layer_finish(layer_id);
    _layer_stat_map[layer_id] = _active_layers_map[layer_id];
    _active_layers_map.erase(layer_id);
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
      std::unique_ptr<Model> finished_model = std::move(_request_queue.front().model);
      _request_queue.pop_front();
      if(finished_model->check_language_model()) {
        static_cast<Simulator*>(_simulator)->finish_language_model(finished_model->get_id());
      }
      if (finished_model->check_regressive()) {
        finished_model->prepare_regressive();
        static_cast<Simulator*>(_simulator)->register_model(std::move(finished_model));
      }
      
    }
  }
  bool all_empty = tile_queue_empty();
  if (!_request_queue.empty() && all_empty &&
      count_active_layers() == 0) {
    spdlog::info("executable layer count {}",
                 _request_queue.front().model->executable_layer_size());
    Operation* new_layer =
        _request_queue.front().model->get_executable_tile();
    /* Check executable layer exist */
    if (new_layer == nullptr)
      return;

    spdlog::info("Start layer {}", new_layer->get_name().c_str());
    _request_queue.front().model->update_start_time(*_core_time);
    /* Get tiles from new layer */
    _executable_tile_queue[0].insert(
        _executable_tile_queue[0].end(),
        std::make_move_iterator(new_layer->get_tiles().begin()),
        std::make_move_iterator(new_layer->get_tiles().end())
    );
    new_layer->clear_tiles();

    _nr_layer++;
    _active_layers_map[new_layer->get_id()] =
        LayerStat{.id = new_layer->get_id(),
                  .name = new_layer->get_name(),
                  .launched = true,
                  .start_cycle = *_core_cycle,
                  .total_tiles = (uint32_t)_executable_tile_queue[0].size(),
                  .remain_tiles = (uint32_t)_executable_tile_queue[0].size(),
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
                                       const cycle_type* core_cycle, const uint64_t* core_time, void* simulator)
    : TimeMultiplexScheduler(config, core_cycle, core_time, simulator) {}

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
        std::unique_ptr<Model> finished_model = std::move(req->model);
        req = _request_queue.erase(req);
        if (finished_model->check_regressive()) {
          finished_model->prepare_regressive();
          static_cast<Simulator*>(_simulator)->register_model(std::move(finished_model));
        }
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
      uint32_t partition_id = cpu_to_partition(i);
      uint32_t offset = 0;
      std::vector<uint32_t> allowed_cpu;
      int req_index;

      if (!_executable_tile_queue[partition_id].empty())
        continue;

      for (req_index=0; req_index<_request_queue.size(); req_index++) {
        /* Skip model requests */
        if (partition_id != _request_queue[req_index].model->get_partition_id())
          continue;

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

        _request_queue[req_index].model->update_start_time(*_core_time);
        _executable_tile_queue[partition_id].insert(
          _executable_tile_queue[partition_id].end(),
          std::make_move_iterator(new_layer->get_tiles().begin()),
          std::make_move_iterator(new_layer->get_tiles().end())
        );
        _nr_layer++;
        _active_layers_map[new_layer->get_id()] =
            LayerStat{.id = new_layer->get_id(),
                      .request_id = _request_queue[req_index].request_id,
                      .name = new_layer->get_name(),
                      .launched = true,
                      .start_cycle = *_core_cycle,
                      .total_tiles = (uint32_t)_executable_tile_queue[partition_id].size(),
                      .remain_tiles = (uint32_t)_executable_tile_queue[partition_id].size(),
                      .finished_tiles = 0,
                      .launched_tiles = 0};
        issue_tile_per_core(allowed_cpu, offset, partition_id);
      } else {
        spdlog::error("Accessed executing layer... id:{}, name:{} total:{} remain:{} fin:{} launched:{}",
        _active_layers_map[new_layer->get_id()].id, _active_layers_map[new_layer->get_id()].name,
        _active_layers_map[new_layer->get_id()].total_tiles, _active_layers_map[new_layer->get_id()].remain_tiles,
        _active_layers_map[new_layer->get_id()].finished_tiles, _active_layers_map[new_layer->get_id()].launched_tiles);
      }
    }
  }
}

TimeMultiplexScheduler::TimeMultiplexScheduler(SimulationConfig config,
                                       const cycle_type* core_cycle, const uint64_t* core_time, void* simulator)
    : Scheduler(config, core_cycle, core_time, simulator) {}

void TimeMultiplexScheduler::finish_tile(uint32_t core_id, int layer_id) {
  spdlog::debug("Layer {} Core {} Finish Tile at {} Remain tile {}", layer_id, core_id,
                *_core_cycle, _active_layers_map[layer_id].remain_tiles);
  assert(_active_layers_map.find(layer_id) != _active_layers_map.end());
  assert(_active_layers_map[layer_id].remain_tiles > 0);
  _active_layers_map[layer_id].remain_tiles--;
  _active_layers_map[layer_id].finished_tiles++;

  if (_active_layers_map[layer_id].remain_tiles == 0) {
    _active_layers_map[layer_id].finish_cycle = *_core_cycle;
    std::string model_name;
    bool model_finish = false;
    for (int req_index = 0; req_index < _request_queue.size(); req_index++) {
      if (_request_queue[req_index].request_id ==
          _active_layers_map[layer_id].request_id) {
        model_finish = true;
        _request_queue[req_index].model->set_layer_finish(layer_id);
        model_name = _request_queue[req_index].model->get_name();
      }
    }
    spdlog::info("Layer {} {} finish at {}", model_name,
                 _active_layers_map[layer_id].name, *_core_cycle);
    spdlog::info("Total compute time {}",
                 *_core_cycle - _active_layers_map[layer_id].start_cycle);
    assert(model_finish);
    _layer_stat_map[layer_id] = _active_layers_map[layer_id];
    _active_layers_map.erase(layer_id);
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
        std::unique_ptr<Model> finished_model = std::move(req->model);
        req = _request_queue.erase(req);
        if (finished_model->check_regressive()) {
          finished_model->prepare_regressive();
          static_cast<Simulator*>(_simulator)->register_model(std::move(finished_model));
        }
        --req;
      }
    }
  }
  bool all_empty = tile_queue_empty();
  if (!_request_queue.empty() && all_empty) {
    _request_rr = (_request_rr + 1) % _request_queue.size();
    Operation* new_layer =
        _request_queue[_request_rr].model->get_executable_tile();
    /* Check executable layer exist */
    if (new_layer == nullptr)
      return;

    if (_active_layers_map.find(new_layer->get_id()) ==
        _active_layers_map.end()) {
      if (count_active_layers() > 0)
        spdlog::info("Layer {} {}: launched before finish prior layer",
                     new_layer->get_name(), new_layer->get_id());
      else
        spdlog::info("Layer {} {}: Enqueue", new_layer->get_name(),
                     new_layer->get_id());

      _request_queue[_request_rr].model->update_start_time(*_core_time);
      _executable_tile_queue[0].insert(
        _executable_tile_queue[0].end(),
        std::make_move_iterator(new_layer->get_tiles().begin()),
        std::make_move_iterator(new_layer->get_tiles().end())
      );
      _nr_layer++;
      _active_layers_map[new_layer->get_id()] =
          LayerStat{.id = new_layer->get_id(),
                    .request_id = _request_queue[_request_rr].request_id,
                    .name = new_layer->get_name(),
                    .launched = true,
                    .start_cycle = *_core_cycle,
                    .total_tiles = (uint32_t)_executable_tile_queue[0].size(),
                    .remain_tiles = (uint32_t)_executable_tile_queue[0].size(),
                    .finished_tiles = 0,
                    .launched_tiles = 0};

      /* Issue tiles to core scheduler */
      issue_tile_per_core();
    }
  }
}

HalfSplitScheduler::HalfSplitScheduler(SimulationConfig config,
                                       const cycle_type* core_cycle, const uint64_t* core_time, void* simulator)
    : Scheduler(config, core_cycle, core_time, simulator) {}

void HalfSplitScheduler::schedule_model(std::unique_ptr<Model> model,
                                        uint32_t sample_size) {
  _request_queue.push_back(Request{.request_id = generate_id(),
                                   .model = std::move(model),
                                   .sample_size = sample_size});
  spdlog::info("MODEL {} Scheduled, Total Request: {}",
               _request_queue.back().model->get_name(), _request_queue.size());
  _executable_tile_queue_table[_request_queue.back().request_id] =
      std::deque<std::unique_ptr<Tile>>();
  refresh_status();
}

std::unique_ptr<Tile> HalfSplitScheduler::get_tile(uint32_t core_id) {
  uint32_t target_id = core_id % _request_queue.size();
  uint32_t req_id = _request_queue[target_id].request_id;
  if (_executable_tile_queue_table[req_id].empty()) {
    std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{});
    tile->status = Tile::Status::EMPTY;
    return tile;
  } else {
    std::unique_ptr<Tile> tile = std::move(_executable_tile_queue_table[req_id].front());
    _executable_tile_queue_table[req_id].pop_front();
    if (!_active_layers_map[tile->layer_id].launched) {
      _active_layers_map[tile->layer_id].launched = true;
      _active_layers_map[tile->layer_id].start_cycle = *_core_cycle;
      spdlog::info("Start layer {}", _active_layers_map[tile->layer_id].name);
    }
    return tile;
  }
}

void HalfSplitScheduler::finish_tile(uint32_t core_id, int layer_id) {
  assert(_active_layers_map.find(layer_id) != _active_layers_map.end());
  assert(_active_layers_map[layer_id].remain_tiles > 0);
  _active_layers_map[layer_id].remain_tiles--;
  _active_layers_map[layer_id].finished_tiles++;

  if (_active_layers_map[layer_id].remain_tiles == 0) {
    _active_layers_map[layer_id].finish_cycle = *_core_cycle;
    std::string model_name;
    bool model_finish = false;
    for (int req_index = 0; req_index < _request_queue.size(); req_index++) {
      if (_request_queue[req_index].request_id ==
          _active_layers_map[layer_id].request_id) {
        model_finish = true;
        _request_queue[req_index].model->set_layer_finish(layer_id);
        model_name = _request_queue[req_index].model->get_name();
        _executable_tile_queue_table.erase(
            _request_queue[req_index].request_id);
      }
    }
    spdlog::info("Layer {} {} finish at {}", model_name,
                 _active_layers_map[layer_id].name, *_core_cycle);
    spdlog::info("Total compute time {}",
                 *_core_cycle - _active_layers_map[layer_id].start_cycle);
    assert(model_finish);
    _layer_stat_map[layer_id] = _active_layers_map[layer_id];
    _active_layers_map.erase(layer_id);
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
        std::unique_ptr<Model> finished_model = std::move(req->model);
        req = _request_queue.erase(req);
        if (finished_model->check_regressive()) {
          finished_model->prepare_regressive();
          static_cast<Simulator*>(_simulator)->register_model(std::move(finished_model));
        }
        --req;
      }
    }
  }
  if (!_request_queue.empty()) {
    for (auto req = _request_queue.begin(); req != _request_queue.end();
         req++) {
      if (_executable_tile_queue_table[req->request_id].empty()) {
        Operation* new_layer = req->model->get_executable_tile();
        /* Check executable layer exist */
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

          auto& tiles = new_layer->get_tiles();
          _executable_tile_queue_table[req->request_id].insert(_executable_tile_queue_table[req->request_id].begin(),
            std::make_move_iterator(tiles.begin()), std::make_move_iterator(tiles.end()));

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
