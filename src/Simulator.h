#pragma once

#include "Common.h"
#include "Core.h"
#include "Dram.h"
#include "Interconnect.h"
#include "Model.h"
#include "scheduler/Scheduler.h"
#include <queue>

#define CORE_MASK 0x1 << 1
#define DRAM_MASK 0x1 << 2
#define ICNT_MASK 0x1 << 3

class Simulator {
 public:
  Simulator(SimulationConfig config);
  void register_model(std::unique_ptr<Model> model);
  void run_simulator();
  const double get_tile_ops();
  const size_t get_number_tile() { return _tile_timestamp.size(); }
  // void run_offline(std::string model_name, uint32_t sample_count);
  // void run_multistream(std::string model_name, uint32_t sample_count,
  // uint32_t ); void run_server(std::string trace_path);
 private:
  void cycle();
  bool running();
  void set_cycle_mask();
  void handle_model();
  uint32_t get_dest_node(MemoryAccess* access);
  SimulationConfig _config;
  uint32_t _n_cores;
  uint32_t _n_memories;

  // Components
  std::vector<std::unique_ptr<Core>> _cores;
  std::unique_ptr<Interconnect> _icnt;
  std::unique_ptr<Dram> _dram;
  std::unique_ptr<Scheduler> _scheduler;
  
  // period information (ps)
  uint64_t _core_period;
  uint64_t _icnt_period;
  uint64_t _dram_period;
  //
  uint64_t _core_time;
  uint64_t _icnt_time;
  uint64_t _dram_time;

  addr_type _dram_ch_stride_size;

  uint64_t _core_cycles;

  uint32_t _cycle_mask;
  bool _single_run;

  struct CompareModel {
    bool operator()(const std::unique_ptr<Model>& a, const std::unique_ptr<Model>& b) const {
        return a->get_request_time() > b->get_request_time();
    }
  };
  robin_hood::unordered_map<std::string, 
    std::vector<std::unique_ptr<Tensor>>> _weight_table;
  std::vector<std::unique_ptr<Model>>  _models;
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> _tile_timestamp;

  bool check_defined_model(std::string model_name);
};