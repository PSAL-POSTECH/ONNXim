#include "Core.h"
#include "SystolicWS.h"
#include "SystolicOS.h"

#include "helper/HelperFunctions.h"

std::unique_ptr<Core> Core::create(uint32_t id, SimulationConfig config) {
  if (config.core_config[id].core_type == CoreType::SYSTOLIC_WS) {
    return std::make_unique<SystolicWS>(id, config);
  } else if (config.core_config[id].core_type == CoreType::SYSTOLIC_OS) {
    return std::make_unique<SystolicOS>(id, config);
  } else {
      spdlog::error("[Configuration] Invalid core type...!");
    exit(EXIT_FAILURE);
  }
}

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_idle_cycle(0),
      _stat_memory_idle_cycle(0),
      _stat_vec_compute_cycle(0),
      _stat_matmul_cycle(0),
      _spad(Sram(config, _core_cycle, false, id)),
      _acc_spad(Sram(config, _core_cycle, true, id)) {
  _waiting_write_reqs = 0;
  _running_layer = -1;
}

bool Core::can_issue(bool is_accum_tile) {
  return _tiles.size() < 2;  // double buffer
}

void Core::issue(std::unique_ptr<Tile> op) {
  op->stat = {.start_cycle = _core_cycle,
             .cycles = 0,
             .compute_cycles = 0,
             .memory_stall = 0,
             .sram_reads = 0,
             .sram_writes = 0};
  int spad_id = 0;
  int acc_spad_id = 0;

  if (_tiles.size() == 1) {
    spad_id = _tiles[0]->spad_id;
    acc_spad_id = _tiles[0]->accum_spad_id;
  }

  /* Double buffer */
  spad_id = (spad_id + 1) % 2;
  _spad.flush(spad_id);
  if (!op->accum || !(_current_layer_id == op->layer_id && _current_fused_op_id == op->fused_op_id)) {
    /* Accumeulate tile uses same acc spad buffer */
    acc_spad_id = (acc_spad_id + 1) % 2;
    _acc_spad.flush(acc_spad_id);
  }
  _current_layer_id = op->layer_id;
  _current_fused_op_id = op->fused_op_id;

  op->spad_id = spad_id;
  op->accum_spad_id = acc_spad_id;
  op->status = Tile::Status::RUNNING;
  if (op->skip) {
    op->status = Tile::Status::FINISH;
    _finished_tiles.push(std::move(op));
    return;
  }
  if (_running_layer != op->layer_id) {
    _running_layer = op->layer_id;
  }
  _tiles.push_back(std::move(op));
}

std::unique_ptr<Tile> Core::pop_finished_tile() {
  std::unique_ptr<Tile> result = std::make_unique<Tile>(Tile{});
  result->status = Tile::Status::EMPTY;
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

void Core::cycle() {
  _core_cycle++;
  _spad.cycle();
  _acc_spad.cycle();
  for (int tile_iter = 0; tile_iter < _tiles.size(); tile_iter++) {
    int i = (tile_iter + tile_rr) % _tiles.size();
    if(_tiles[i]->instructions.empty()) 
      continue;
    std::unique_ptr<Instruction>& inst = _tiles[i]->instructions.front();
    if(_tiles[i]->instructions.size() == 1) {
      inst->last_inst = true;
      inst->my_tile = _tiles[i].get();
    }
    inst->spad_id = _tiles[i]->spad_id;
    inst->accum_spad_id = _tiles[i]->accum_spad_id;
    Sram *buffer;
    int buffer_id;
    if (inst->dest_addr >= ACCUM_SPAD_BASE) {
      buffer = &_acc_spad;
      buffer_id = _tiles[i]->accum_spad_id;
    } else {
      buffer = &_spad;
      buffer_id = _tiles[i]->spad_id;
    }
    bool issued = false;
    if (inst->opcode == Opcode::MOVIN) {
      /*LD inst queue */
      if (inst->size == 0) {
        spdlog::error("[Core {}] MVIN issue addr: {:x}, size: {:x}", _id, inst->dest_addr, inst->size);
      }
      if (!buffer->check_allocated(inst->dest_addr, buffer_id) &&
          buffer->check_remain(inst->size, buffer_id)) {
        _ld_inst_queue.push(std::move(inst));
        issued = true;
      } else {
        /*Invalid state */
        spdlog::error("Destination allocated: {} Size remain: {}", buffer->check_allocated(inst->dest_addr, buffer_id), buffer->check_remain(inst->size, buffer_id));
        spdlog::error("[Core {}] MVIN issue panic addr: {:x}, size: {} B", _id, inst->dest_addr, inst->size*_config.dram_req_size);
        buffer->print_all(buffer_id);
        exit(EXIT_FAILURE);
      }
    } else if (inst->opcode == Opcode::MOVOUT ||
               inst->opcode == Opcode::MOVOUT_POOL) {
      /* ST inst queue */
      if (buffer->check_hit(inst->dest_addr, buffer_id)) {
        _st_inst_queue.push(std::move(inst));
        issued = true;
      }
    } else {
      /* Ex inst queue */
      if (inst.get() == 0) {
        spdlog::error("null instruction!");
      }
      if(_ex_inst_queue.empty() && can_issue_compute(inst)) {
        _ex_inst_queue.push(std::move(inst));
        issued = true;
      }
    }
    if (issued) {
      _tiles[i]->instructions.pop_front();
      tile_rr = i;
      break;
    }
  }
  for (auto tile = _tiles.begin() ; tile < _tiles.end(); tile++) {
    if ((*tile)->instructions.empty() && (*tile)->inst_finished) {
      (*tile)->status = Tile::Status::FINISH;
      (*tile)->stat.cycles = _core_cycle - (*tile)->stat.start_cycle;
      (*tile)->stat.memory_stall =
          (*tile)->stat.cycles - (*tile)->stat.compute_cycles;
      _finished_tiles.push(std::move(*tile));
      _tiles.erase(tile);
      break;
    }
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_compute_pipeline.empty();
  running = running ||
            !_vector_pipeline.empty();  // Vector unit (Might need to modify)
  running = running || _waiting_write_reqs != 0;
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  running = running || !_ex_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() { return _request_queue.size() > 0; }

void Core::pop_memory_request() {
  assert(has_memory_request());
  _request_queue.pop();
}

void Core::push_memory_response(MemoryAccess *response) {
  assert(!response->request);  // can only push response
  if (response->write) {
    _waiting_write_reqs--;
  } else if (response->spad_address >= ACCUM_SPAD_BASE) {
    _acc_spad.fill(response->spad_address, response->buffer_id);
  } else {
    assert(_spad.check_allocated(response->spad_address, response->buffer_id));
    _spad.fill(response->spad_address, response->buffer_id);
  }
  delete response;
}

bool Core::can_issue_compute(std::unique_ptr<Instruction>& inst) {
  bool result = true;

  for (addr_type addr : inst->src_addrs) {
    if (inst->src_from_accum && addr >= ACCUM_SPAD_BASE) {
      result = result && _acc_spad.check_hit(addr, inst->accum_spad_id);
    } else {
      result = result && _spad.check_hit(addr, inst->spad_id);
    }
  }
  if (!result) {
    for (addr_type addr : inst->src_addrs) {
      spdlog::trace("Core[{}] Dependency fail : {} , {}", _id, addr,
                    _spad.check_hit(addr, inst->spad_id));
    }
  }
  return result;
}

void Core::print_stats() {
  update_stats();
  spdlog::info(
      "Core [{}] : MatMul active cycle {} Vector active cycle {} ",
      _id, _stat_tot_matmul_cycle, _stat_tot_vec_compute_cycle);

  spdlog::info(
      "Core [{}] : Memory unit idle cycle {} Systolic bubble cycle {} "
      "Core idle cycle {} ",
      _id, _stat_tot_memory_idle_cycle, _stat_tot_systolic_bubble_cycle, _stat_tot_idle_cycle);

  spdlog::info("Core [{}] : Systolic Array Utilization(%) {:.2f} ({:.2f}% PE util), Vector Unit Utilization(%) {:.2f}, Total cycle: {}",
      _id, static_cast<float>(_stat_tot_systolic_active_cycle * 100) / _core_cycle,
      static_cast<float>(_stat_tot_matmul_cycle * 100) / _core_cycle,
      static_cast<float>(_stat_tot_vec_compute_cycle * 100) / _core_cycle, _core_cycle);
}

void Core::print_current_stats() {
  auto level = spdlog::level::info;
  if(_id != 0) 
    level = spdlog::level::debug;
    spdlog::log(level,
      "Core [{}] : MatMul active cycle {} Vector active cycle {} ",
      _id, _stat_matmul_cycle, _stat_vec_compute_cycle);

  spdlog::log(level,
      "Core [{}] : issued tile {} ", _id, _tiles.size());

  spdlog::log(level,
      "Core [{}] : Memory unit idle cycle {} Systolic bubble cycle {} "
      "Core idle cycle {} ",
      _id, _stat_memory_idle_cycle, _stat_systolic_bubble_cycle, _stat_idle_cycle);
  spdlog::log(level,"Core [{}] : Systolic Array Utilization(%) {:.2f} ({:.2f}% PE util), Vector Unit Utilization(%) {:.2f}, Total cycle: {}",
      _id, static_cast<float>(_stat_systolic_active_cycle * 100) / _config.core_print_interval,
      static_cast<float>(_stat_matmul_cycle * 100) / _config.core_print_interval,
      static_cast<float>(_stat_vec_compute_cycle * 100) / _config.core_print_interval, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  _stat_tot_compute_cycle += _stat_compute_cycle;
  _stat_tot_systolic_active_cycle += _stat_systolic_active_cycle;
  _stat_tot_systolic_bubble_cycle += _stat_systolic_bubble_cycle;
  _stat_tot_memory_idle_cycle += _stat_memory_idle_cycle;
  _stat_tot_idle_cycle += _stat_idle_cycle;
  _stat_tot_vec_compute_cycle += _stat_vec_compute_cycle;
  _stat_tot_matmul_cycle += _stat_matmul_cycle;
  _stat_compute_cycle = 0;
  _stat_systolic_active_cycle = 0;
  _stat_systolic_bubble_cycle = 0;
  _stat_memory_idle_cycle = 0;
  _stat_idle_cycle = 0;
  _stat_vec_compute_cycle = 0;
  _stat_matmul_cycle = 0;
}

void Core::finish_compute_pipeline(){
  if (!_compute_pipeline.empty() &&
      _compute_pipeline.front()->finish_cycle <= _core_cycle) {
    std::unique_ptr<Instruction> inst = std::move(_compute_pipeline.front());
    if (inst->dest_addr >= ACCUM_SPAD_BASE)
      _acc_spad.fill(inst->dest_addr, inst->accum_spad_id);
    else
      _spad.fill(inst->dest_addr, inst->spad_id);
    if(inst->last_inst) {
      spdlog::trace("Finished last GEMM {}", inst->spad_id);
      inst->my_tile->inst_finished = true;
    }
    double compute_size = inst->tile_k * inst->tile_m * inst->tile_n
                            / (_config.core_config[_id].core_height * _config.core_config[_id].core_width);
    spdlog::trace("Compute size {} tile m {} tile k {} tile n {}", inst->compute_size, inst->tile_m, inst->tile_k, inst->tile_n);
    spdlog::trace("Compute size {} , compute time {}", compute_size, inst->finish_cycle - inst->start_cycle);
    _stat_matmul_cycle += compute_size;
    _compute_pipeline.pop();
  }
}

void Core::finish_vector_pipeline() {
  if (!_vector_pipeline.empty() &&
      _vector_pipeline.front()->finish_cycle <= _core_cycle) {
    std::unique_ptr<Instruction> inst = std::move(_vector_pipeline.front());
    if (inst->dest_addr >= ACCUM_SPAD_BASE) {
      if(!_acc_spad.check_allocated(inst->dest_addr, inst->accum_spad_id)) {
        spdlog::error("Vector pipeline -> accum");
        spdlog::error("Destination not allocated {}", inst->dest_addr);
      }
      _acc_spad.fill(inst->dest_addr, inst->accum_spad_id);
    }
    else {
      if(!_spad.check_allocated(inst->dest_addr, inst->accum_spad_id)) {
        spdlog::error("Vector pipeline -> spad");
        spdlog::error("Destination not allocated {}", inst->dest_addr);
      }
      _spad.fill(inst->dest_addr, inst->spad_id);
    }
      
    if(inst->last_inst)
      inst->my_tile->inst_finished = true;
    _vector_pipeline.pop();
  }
}

void Core::handle_ld_inst_queue() {
  if (!_ld_inst_queue.empty()) {
    std::unique_ptr<Instruction> front = std::move(_ld_inst_queue.front());
    if (front->opcode == Opcode::MOVIN) {
      bool prefetched = false;
      Sram *buffer;
      int buffer_id;
      if (front->dest_addr >= ACCUM_SPAD_BASE) {
        buffer = &_acc_spad;
        buffer_id = front->accum_spad_id;
      } else {
        buffer = &_spad;
        buffer_id = front->spad_id;
      }
      if (front->size==0) {
        spdlog::error("Destination size is 0! opcode: {}, addr: 0x{:x}", (int)front->opcode, front->dest_addr);
      }
      int ret = buffer->prefetch(front->dest_addr, buffer_id, front->size, front->size);
      if (!ret) {
        spdlog::error("Destination allocated: {} Size remain: {}", buffer->check_allocated(front->dest_addr, buffer_id), buffer->check_remain(front->size, buffer_id));
        spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
        std::exit(EXIT_FAILURE);
      }
      for (addr_type addr : front->src_addrs) {
        assert(front->base_addr != GARBEGE_ADDR);
        MemoryAccess *access =
            new MemoryAccess({.id = generate_mem_access_id(),
                              .dram_address = addr + front->base_addr,
                              .spad_address = front->dest_addr,
                              .size = _config.dram_req_size,
                              .write = false,
                              .request = true,
                              .core_id = _id,
                              .start_cycle = _core_cycle,
                              .buffer_id = buffer_id});
        _request_queue.push(access);
      }
      _ld_inst_queue.pop();
    } else {
      assert(0);
    }
  }
}

void Core::handle_st_inst_queue() {
  if (!_st_inst_queue.empty()) {
    std::unique_ptr<Instruction> front = std::move(_st_inst_queue.front());
    if (front->opcode == Opcode::MOVOUT || front->opcode == Opcode::MOVOUT_POOL) {
      Sram *buffer;
      int buffer_id;
      if (front->dest_addr >= ACCUM_SPAD_BASE) {
        buffer = &_acc_spad;
        buffer_id = front->accum_spad_id;
      } else {
        buffer = &_spad;
        buffer_id = front->spad_id;
      }
      if(buffer->check_hit(front->dest_addr, buffer_id)) {
        for (addr_type addr : front->src_addrs) {
          assert(front->base_addr != GARBEGE_ADDR);
          MemoryAccess *access =
              new MemoryAccess{.id = generate_mem_access_id(),
                              .dram_address = addr + front->base_addr,
                              .spad_address = front->dest_addr,
                              .size = _config.dram_req_size,
                              .write = true,
                              .request = true,
                              .core_id = _id,
                              .start_cycle = _core_cycle,
                              .buffer_id = buffer_id};
          _waiting_write_reqs++;
          _request_queue.push(access);
        }
        if(front->last_inst) {
          spdlog::trace("Finished last store {}", front->spad_id);
          front->my_tile->inst_finished = true;
        }
        _st_inst_queue.pop();
      }
    } else {
      assert(0);
    }
  }
}

cycle_type Core::calculate_add_tree_iterations(uint32_t vector_size) {
  uint32_t calculation_unit = _config.core_config[_id].vector_process_bit >> 3;
  if (vector_size <= calculation_unit) {
    return 1;
  }

  uint32_t ret = vector_size / calculation_unit;
  if (vector_size % calculation_unit != 0) {
    ret++;
  }
  return ret + calculate_add_tree_iterations(ret);
}

cycle_type Core::calculate_vector_op_iterations(uint32_t vector_size) {
  uint32_t calculation_unit = _config.core_config[_id].vector_process_bit >> 3;
  uint32_t ret = vector_size / calculation_unit;
  if (vector_size % calculation_unit != 0) {
    ret++;
  }
  return ret;
}