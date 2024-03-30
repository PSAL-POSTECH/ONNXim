#pragma once
#include <robin_hood.h>

#include <memory>
#include <vector>

#include "Dram.h"
#include "SimulationConfig.h"
#include "Sram.h"
#include "Stat.h"

class Core {
 public:
  Core(uint32_t id, SimulationConfig config);
  virtual ~Core() = default;
  virtual bool running();
  virtual bool can_issue(bool is_accum_tile=false);
  virtual void issue(Tile tile);
  virtual Tile pop_finished_tile();

  virtual void cycle();

  virtual bool has_memory_request();
  virtual void pop_memory_request();
  virtual MemoryAccess* top_memory_request() { return _request_queue.front(); }
  virtual void push_memory_response(MemoryAccess* response);
  virtual void print_stats();
  virtual cycle_type get_compute_cycles() { return _stat_compute_cycle; }

 protected:
  virtual bool can_issue_compute(Instruction inst);
  virtual cycle_type get_inst_compute_cycles(Instruction inst) = 0;

  const uint32_t _id;
  const SimulationConfig _config;

  cycle_type _core_cycle;
  uint64_t _compute_end_cycle;
  cycle_type _stat_compute_cycle;
  cycle_type _stat_idle_cycle;
  cycle_type _stat_memory_cycle;
  cycle_type _accum_request_rr_cycle;
  cycle_type _max_request_rr_cycle;
  cycle_type _min_request_rr_cycle;

  cycle_type _compute_memory_stall_cycle;
  cycle_type _layernorm_stall_cycle;
  cycle_type _softmax_stall_cycle;
  cycle_type _add_stall_cycle;
  cycle_type _gelu_stall_cycle;
  cycle_type _load_memory_cycle;
  cycle_type _store_memory_cycle;

  /* Vector Unit Params */
  cycle_type _stat_vec_compute_cycle;
  cycle_type _stat_vec_memory_cycle;  // Does not acctuall count yet
  cycle_type _stat_vec_idle_cycle;    // Does not acctuall count yet

  cycle_type _stat_matmul_cycle;
  cycle_type _stat_layernorm_cycle;
  cycle_type _stat_add_cycle;
  cycle_type _stat_gelu_cycle;
  cycle_type _stat_softmax_cycle;

  int _running_layer;
  std::deque<Tile> _tiles;
  std::queue<Tile> _finished_tiles;

  std::queue<Instruction> _compute_pipeline;
  std::queue<Instruction> _vector_pipeline;

  std::queue<Instruction> _ld_inst_queue;
  std::queue<Instruction> _st_inst_queue;
  std::queue<Instruction> _ex_inst_queue;

  std::queue<MemoryAccess*> _request_queue;
  std::queue<MemoryAccess*> _response_queue;
  uint32_t _waiting_write_reqs;

  int _current_spad;
  int _current_acc_spad;
  uint32_t _current_layer_id;
  uint32_t _current_fused_op_id;
  Sram _spad;
  Sram _acc_spad;
};