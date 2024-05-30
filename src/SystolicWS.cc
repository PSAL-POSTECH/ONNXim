#include "SystolicWS.h"

SystolicWS::SystolicWS(uint32_t id, SimulationConfig config)
    : Core(id, config) {}

void SystolicWS::cycle() {
  /*
  Compute unit
  */
  if (!_compute_pipeline.empty() &&
      _compute_pipeline.front()->finish_cycle <= _core_cycle) {
    std::unique_ptr<Instruction> inst = std::move(_compute_pipeline.front());
    if (inst->dest_addr >= ACCUM_SPAD_BASE)
      _acc_spad.fill(inst->dest_addr, inst->accum_spad_id);
    else
      _spad.fill(inst->dest_addr, inst->spad_id);
    if(inst->last_inst)
      inst->my_tile->inst_finished = true;
    _compute_pipeline.pop();
  }

  /* Checking Vector compute pipeline */
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
  /* LD in struction queue */
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
        spdlog::error("Destination allocated: {} Size remain: {}", buffer->check_allocated(front->dest_addr, buffer_id), buffer->check_allocated(front->dest_addr, buffer_id));
        spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {:x}", (int)front->opcode, front->dest_addr, front->size);
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

  /* EX instruction queue */
  if (!_ex_inst_queue.empty() && can_issue_compute(_ex_inst_queue.front())) { // execution dependecy check
    std::unique_ptr<Instruction> front = std::move(_ex_inst_queue.front());
    if (front->dest_addr >= ACCUM_SPAD_BASE) {
      if (_acc_spad.check_allocated(front->dest_addr, front->accum_spad_id)) {
        _acc_spad.count_up(front->dest_addr, front->accum_spad_id);
      } else {
        int ret = _acc_spad.prefetch(front->dest_addr, front->accum_spad_id, front->size, front->zero_init? front->size : 1);
        if (!ret) {
          spdlog::error("Destination allocated: {} Size remain: {}", _acc_spad.check_allocated(front->dest_addr, front->accum_spad_id), _acc_spad.check_allocated(front->dest_addr, front->accum_spad_id));
          spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {:x}", (int)front->opcode, front->dest_addr, front->size*32);
          _acc_spad.print_all(front->accum_spad_id);
          std::exit(EXIT_FAILURE);
        }
      }
    } else {
      if (_spad.check_allocated(front->dest_addr, front->spad_id)) {
        _spad.count_up(front->dest_addr, front->spad_id);
      } else {
        int ret = _spad.prefetch(front->dest_addr, front->spad_id, front->size, front->zero_init? front->size : 1);
        if (!ret) {
          spdlog::error("Destination allocated: {} Size remain: {}", _spad.check_allocated(front->dest_addr, front->spad_id), _spad.check_allocated(front->dest_addr, front->spad_id));
          spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {:x}", (int)front->opcode, front->dest_addr, front->size*32);
          _spad.print_all(front->spad_id);
          std::exit(EXIT_FAILURE);
        }
      }
    }
    if (front->opcode == Opcode::GEMM || front->opcode == Opcode::GEMM_PRELOAD) {
      if (!_compute_pipeline.empty()) {
        /* Preload can be hided */
        uint32_t offset = _compute_pipeline.back()->compute_size;
        offset = MAX(offset, 4);
        if (front->opcode == Opcode::GEMM_PRELOAD) {
          // State mul-pre
          offset = _config.core_height;
          _stat_systolic_preload_issue_count++;
        }
        front->start_cycle = _compute_pipeline.back()->start_cycle + offset;
      } else {
        front->start_cycle = _core_cycle;
        /* Preload weight to systolic array*/
        if (front->opcode == Opcode::GEMM_PRELOAD) {
          /* Weight preload  from buffer latecny + WEight preload latency */
          front->start_cycle += _config.core_height + _config.core_height - 1;
          _stat_systolic_preload_issue_count++;
        }
      }

      front->finish_cycle = front->start_cycle + get_inst_compute_cycles(front);
      _compute_pipeline.push(std::move(front));
      _stat_systolic_inst_issue_count++;
    } else if (front->opcode == Opcode::COMP || front->opcode == Opcode::SOFTMAX ||
               front->opcode == Opcode::IM2COL || front->opcode == Opcode::LAYERNORM ||
               front->opcode == Opcode::ADD || front->opcode == Opcode::MUL ||
               front->opcode == Opcode::GELU || front->opcode == Opcode::SWISH) {  // vector unit compute
      if (!_vector_pipeline.empty()) {
        front->start_cycle =
            _vector_pipeline.back()->start_cycle + _vector_pipeline.back()->size;
      } else {
        front->start_cycle = _core_cycle;
      }
      front->finish_cycle =
          front->start_cycle +
          get_vector_compute_cycles(front);  // Setting IC as 1 (Might need to modify)
      _vector_pipeline.push(std::move(front));

    }
    _ex_inst_queue.pop();
  }

  /* ST in struction queue */
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
        if(front->last_inst) 
          front->my_tile->inst_finished = true;
        _st_inst_queue.pop();
      }
    } else {
      assert(0);
    }
  }

  // xxx will it work well on double buffered code? no.
  bool is_idle = _compute_pipeline.empty() && _vector_pipeline.empty();
  bool is_running = running();

  if (!_compute_pipeline.empty())
    _stat_matmul_cycle++;
  if (!_vector_pipeline.empty()) {
    _stat_vec_compute_cycle++;
    switch (_vector_pipeline.front()->opcode) {
      case Opcode::LAYERNORM:
        _stat_layernorm_cycle++;
        break;
      case Opcode::SOFTMAX:
        _stat_softmax_cycle++;
        break;
      case Opcode::ADD:
        _stat_add_cycle++;
        break;
      case Opcode::GELU:
        _stat_gelu_cycle++;
        break;
    }
  }

  if (_request_queue.empty())
    _stat_memory_idle_cycle++;

  if (!is_running)
    _stat_idle_cycle++;
  Core::cycle();
}

cycle_type SystolicWS::get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) {
  return _config.core_height + _config.core_width - 2 + MAX(inst->compute_size, 4);
}

cycle_type SystolicWS::calculate_add_tree_iterations(uint32_t vector_size) {
  uint32_t calculation_unit = _config.vector_process_bit >> 3;
  if (vector_size <= calculation_unit) {
    return 1;
  }

  uint32_t ret = vector_size / calculation_unit;
  if (vector_size % calculation_unit != 0) {
    ret++;
  }
  return ret + calculate_add_tree_iterations(ret);
}

cycle_type SystolicWS::calculate_vector_op_iterations(uint32_t vector_size) {
  uint32_t calculation_unit = _config.vector_process_bit >> 3;
  uint32_t ret = vector_size / calculation_unit;
  if (vector_size % calculation_unit != 0) {
    ret++;
  }
  return ret;
}

cycle_type SystolicWS::get_vector_compute_cycles(std::unique_ptr<Instruction>& inst) {
  cycle_type vec_op_iter = calculate_vector_op_iterations(inst->compute_size);
  cycle_type add_tree_iter = calculate_add_tree_iterations(inst->compute_size);
  cycle_type add_tree, scalar_ops, vector_ops;
  switch (inst->opcode) {
    case Opcode::LAYERNORM:
      add_tree = 2 * add_tree_iter * _config.add_tree_latency;
      scalar_ops = 2 * _config.scalar_mul_latency + _config.scalar_sqrt_latency;
      // 1 addition, 1 subtraction, 1 division, 2 multiplication.
      vector_ops = vec_op_iter * (2 * _config.add_latency + 3 * _config.mul_latency) * inst->tile_m;
      return add_tree + scalar_ops + vector_ops;
    case Opcode::SOFTMAX:
      // 1 add tree, 1 compare tree
      add_tree = 2 * add_tree_iter * _config.add_tree_latency * inst->tile_m;
      vector_ops =
        vec_op_iter * (_config.add_latency + _config.exp_latency + _config.mul_latency);
      return add_tree + vector_ops;
    case Opcode::ADD:
      return vec_op_iter * _config.add_latency;
    case Opcode::MUL:
      return vec_op_iter * _config.mul_latency;
    case Opcode::SWISH: //TODO: Implement SWISH
    case Opcode::GELU:
      return vec_op_iter * _config.gelu_latency;
    case Opcode::COMP:
      return vec_op_iter * 1;
    
  }
  spdlog::info("not configured operation. {}", inst->id);
  // assert(0);
  return 0;
}

void SystolicWS::print_stats() {
  Core::print_stats();
  spdlog::info("Core [{}] : Systolic Inst Issue Count : {}", _id,
               _stat_systolic_inst_issue_count);
  spdlog::info("Core [{}] : Systolic PRELOAD Issue Count : {}", _id,
               _stat_systolic_preload_issue_count);
}