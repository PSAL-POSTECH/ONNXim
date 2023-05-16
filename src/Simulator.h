#pragma once

#include "Common.h"
#include "Core.h"
#include "Dram.h"
#include "Interconnect.h"
#include "Model.h"
#include "scheduler/Scheduler.h"

#define CORE_MASK 0x1 << 1
#define DRAM_MASK 0x1 << 2
#define ICNT_MASK 0x1 << 3

class Simulator {
 public:
  Simulator(SimulationConfig config);
  void launch_model(std::unique_ptr<Model> model);
  void run_once(std::string model_name);
  void run_models(std::vector<std::string> models);
  // void run_offline(std::string model_name, uint32_t sample_count);
  // void run_multistream(std::string model_name, uint32_t sample_count,
  // uint32_t ); void run_server(std::string trace_path);
 private:
  void cycle();
  bool running();
  void set_cycle_mask();
  uint32_t get_dest_node(MemoryAccess* access);
  SimulationConfig _config;
  uint32_t _n_cores;
  uint32_t _n_memories;

  // Components
  std::vector<std::unique_ptr<Core>> _cores;
  std::unique_ptr<Interconnect> _icnt;
  std::unique_ptr<Dram> _dram;
  std::unique_ptr<Scheduler> _scheduler;
  
  // period information (us)
  double _core_period;
  double _icnt_period;
  double _dram_period;
  //
  double _core_time;
  double _icnt_time;
  double _dram_time;

  addr_type _dram_ch_stride_size;

  uint64_t _core_cycles;

  uint32_t _cycle_mask;
  bool _single_run;
  std::map<std::string, std::unique_ptr<Model>> _models;
};