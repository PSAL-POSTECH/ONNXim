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
  static std::unique_ptr<Core> create(uint32_t id, SimulationConfig config);
  Core(uint32_t id, SimulationConfig config);
  virtual ~Core() = default;
  virtual bool running();
  virtual bool can_issue(bool is_accum_tile=false);
  virtual void issue(std::unique_ptr<Tile> tile);
  virtual std::unique_ptr<Tile> pop_finished_tile();

  virtual void cycle();

  virtual bool has_memory_request();
  virtual void pop_memory_request();
  virtual MemoryAccess* top_memory_request() { return _request_queue.front(); }
  virtual void push_memory_response(MemoryAccess* response);
  virtual void print_stats();
  virtual void print_current_stats();

  virtual cycle_type get_compute_cycles() { return _stat_tot_compute_cycle; }

 protected:
  virtual bool can_issue_compute(std::unique_ptr<Instruction>& inst);
  virtual cycle_type get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) = 0;
  virtual void update_stats();
  virtual void finish_compute_pipeline();
  virtual void finish_vector_pipeline();
  virtual void handle_ld_inst_queue();
  virtual void handle_st_inst_queue();
  virtual cycle_type calculate_add_tree_iterations(uint32_t vector_size);
  virtual cycle_type calculate_vector_op_iterations(uint32_t vector_size);

  const uint32_t _id;
  const SimulationConfig _config;

  cycle_type _core_cycle;
  
  cycle_type _stat_idle_cycle;
  cycle_type _stat_memory_idle_cycle;

  cycle_type _stat_tot_compute_cycle = 0;
  cycle_type _stat_tot_idle_cycle = 0;
  cycle_type _stat_tot_memory_idle_cycle = 0;

  cycle_type _accum_request_rr_cycle;
  cycle_type _max_request_rr_cycle;
  cycle_type _min_request_rr_cycle;
  
  /* Vector Unit Params */
  cycle_type _stat_vec_compute_cycle;
  cycle_type _stat_tot_vec_compute_cycle = 0;

  cycle_type _stat_matmul_cycle;
  cycle_type _stat_tot_matmul_cycle = 0;

  int _running_layer;
  uint32_t tile_rr = 0;
  std::deque<std::unique_ptr<Tile>> _tiles;
  std::queue<std::unique_ptr<Tile>> _finished_tiles;

  std::queue<std::unique_ptr<Instruction>> _compute_pipeline;
  std::queue<std::unique_ptr<Instruction>> _vector_pipeline;

  std::queue<std::unique_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _st_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _ex_inst_queue;

  std::queue<MemoryAccess*> _request_queue;
  std::queue<MemoryAccess*> _response_queue;
  uint32_t _waiting_write_reqs;

  uint32_t _current_layer_id;
  uint32_t _current_fused_op_id;
  Sram _spad;
  Sram _acc_spad;
};