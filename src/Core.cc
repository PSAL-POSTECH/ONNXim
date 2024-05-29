#include "Core.h"

#include "helper/HelperFunctions.h"

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _compute_end_cycle(0),
      _stat_idle_cycle(0),
      _stat_compute_cycle(0),
      _stat_memory_cycle(0),
      _stat_vec_compute_cycle(0),
      _stat_vec_memory_cycle(0),
      _stat_vec_idle_cycle(0),
      _compute_memory_stall_cycle(0),
      _layernorm_stall_cycle(0),
      _softmax_stall_cycle(0),
      _add_stall_cycle(0),
      _gelu_stall_cycle(0),
      _load_memory_cycle(0),
      _store_memory_cycle(0),
      _stat_matmul_cycle(0),
      _stat_layernorm_cycle(0),
      _stat_add_cycle(0),
      _stat_gelu_cycle(0),
      _stat_softmax_cycle(0),
      _spad(Sram(config, _core_cycle, false)),
      _acc_spad(Sram(config, _core_cycle, true)) {
  _waiting_write_reqs = 0;
  _running_layer = -1;
  _current_spad = 0;
  _current_acc_spad = 0;
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
  /* Double buffer */
  _current_spad = (_current_spad + 1) % 2;
  _spad.flush(_current_spad);
  op->spad_id = _current_spad;
  if (!op->accum || !(_current_layer_id == op->layer_id && _current_fused_op_id == op->fused_op_id)) {
    /* Accumeulate tile uses same acc spad buffer */
    _current_acc_spad = (_current_acc_spad + 1) % 2;
    _acc_spad.flush(_current_acc_spad);
  }
  _current_layer_id = op->layer_id;
  _current_fused_op_id = op->fused_op_id;

  op->accum_spad_id = _current_acc_spad;
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
  for (int i = 0; i < _tiles.size(); i++) {
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
        spdlog::error("Destination allocated: {} Size remain: {}", buffer->check_allocated(inst->dest_addr, buffer_id), buffer->check_allocated(inst->dest_addr, buffer_id));
        spdlog::error("[Core {}] MVIN issue panic addr: {:x}, size: {:x}", _id, inst->dest_addr, inst->size);
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
      if(_ex_inst_queue.empty()){
        _ex_inst_queue.push(std::move(inst));
        issued = true;
      }
    }
    if (issued) {
      _tiles[i]->instructions.pop_front();
      break;
    }
  }
  for (int i =0 ; i < _tiles.size(); i++) {
    if (_tiles[i]->instructions.empty() && _tiles[i]->inst_finished) {
      _tiles[i]->status = Tile::Status::FINISH;
      _tiles[i]->stat.cycles = _core_cycle - _tiles[i]->stat.start_cycle;
      _tiles[i]->stat.memory_stall =
          _tiles[i]->stat.cycles - _tiles[i]->stat.compute_cycles;
      _finished_tiles.push(std::move(_tiles[i]));
      _tiles.pop_front();
    }
  }
  if(_core_cycle % _config.core_print_interval == 0) {
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
      "Core [{}] : MatMul cycle {} LayerNorm cycle {} Softmax cycle {} "
      "Add cycle {}  Gelu cycle {}",
      _id, _stat_tot_matmul_cycle, _stat_tot_layernorm_cycle, _stat_tot_softmax_cycle, _stat_tot_add_cycle,
      _stat_tot_gelu_cycle);

  spdlog::info(
      "Core [{}] : MatMul stall cycle {} LayerNorm stall cycle {} "
      "Softmax stall cycle {} Add stall cycle {} Gelu stall cycle {}",
      _id, _stat_tot_compute_memory_stall_cycle, _stat_tot_layernorm_stall_cycle, _stat_tot_softmax_stall_cycle,
      _stat_tot_add_stall_cycle, _stat_tot_gelu_stall_cycle);


  spdlog::info(
      "Core [{}] : Load stall cycle {} Store stall cycle {} "
      "Total memory stall {} Idle cycle {}",
      _id, _stat_tot_load_memory_cycle, _stat_tot_store_memory_cycle,
      _stat_tot_memory_cycle, _stat_tot_idle_cycle);

  spdlog::info("Core [{}] : Total cycle: {}", _id, _core_cycle);
}

void Core::print_current_stats() {
  auto level = spdlog::level::info;
  if(_id != 0) 
    level = spdlog::level::debug;
    spdlog::log(level,
      "Core [{}] : MatMul cycle {} LayerNorm cycle {} Softmax cycle {} "
      "Add cycle {}  Gelu cycle {}",
      _id, _stat_matmul_cycle, _stat_layernorm_cycle, _stat_softmax_cycle, _stat_add_cycle,
      _stat_gelu_cycle);

  spdlog::log(level,
      "Core [{}] : MatMul stall cycle {} LayerNorm stall cycle {} "
      "Softmax stall cycle {} Add stall cycle {} Gelu stall cycle {}",
      _id, _compute_memory_stall_cycle, _layernorm_stall_cycle, _softmax_stall_cycle,
      _add_stall_cycle, _gelu_stall_cycle);

  spdlog::log(level,
      "Core [{}] : Load stall cycle {} Store stall cycle {} "
      "Total memory stall {} Idle cycle {}",
      _id, _load_memory_cycle, _store_memory_cycle,
      _stat_memory_cycle, _stat_idle_cycle);

  spdlog::log(level,"Core [{}] : Total cycle: {}", _id, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  _stat_tot_compute_cycle += _stat_compute_cycle;
  _stat_tot_memory_cycle += _stat_memory_cycle;
  _stat_tot_idle_cycle += _stat_idle_cycle;
  _stat_tot_compute_memory_stall_cycle += _compute_memory_stall_cycle;
  _stat_tot_layernorm_stall_cycle += _layernorm_stall_cycle;
  _stat_tot_softmax_stall_cycle += _softmax_stall_cycle;
  _stat_tot_add_stall_cycle += _add_stall_cycle;
  _stat_tot_gelu_stall_cycle += _gelu_stall_cycle;
  _stat_tot_load_memory_cycle += _load_memory_cycle;
  _stat_tot_store_memory_cycle += _store_memory_cycle;
  _stat_tot_vec_compute_cycle += _stat_vec_compute_cycle;
  _stat_tot_vec_memory_cycle += _stat_vec_memory_cycle;
  _stat_tot_vec_idle_cycle += _stat_vec_idle_cycle;
  _stat_tot_matmul_cycle += _stat_matmul_cycle;
  _stat_tot_layernorm_cycle += _stat_layernorm_cycle;
  _stat_tot_add_cycle += _stat_add_cycle;
  _stat_tot_gelu_cycle += _stat_gelu_cycle;
  _stat_tot_softmax_cycle += _stat_softmax_cycle;
  _stat_compute_cycle = 0;
  _stat_memory_cycle = 0;
  _stat_idle_cycle = 0;
  _compute_memory_stall_cycle = 0;
  _layernorm_stall_cycle = 0;
  _softmax_stall_cycle = 0;
  _add_stall_cycle = 0;
  _gelu_stall_cycle = 0;
  _load_memory_cycle = 0;
  _store_memory_cycle = 0;
  _stat_vec_compute_cycle = 0;
  _stat_vec_memory_cycle = 0;
  _stat_vec_idle_cycle = 0;
  _stat_matmul_cycle = 0;
  _stat_layernorm_cycle = 0;
  _stat_add_cycle = 0;
  _stat_gelu_cycle = 0;
  _stat_softmax_cycle = 0;
}