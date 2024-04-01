#pragma once
#include <robin_hood.h>
#include "../Common.h"
#include "../Model.h"

typedef struct {
  uint32_t request_id;
  std::unique_ptr<Model> model;
  uint32_t sample_size;
} Request;

class Scheduler {
  public:
    Scheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time);
    virtual void schedule_model(std::unique_ptr<Model> model, uint32_t sampe_size);
    virtual std::unique_ptr<Tile> get_tile(uint32_t core_id);
    virtual void issue_tile_per_core();
    virtual void issue_tile_per_core(std::vector<uint32_t>& allowed_cpu, int offset, uint32_t partition_id);
    virtual bool is_accum_tile(uint32_t core_id, int index);
    virtual void finish_tile(uint32_t core_id, std::unique_ptr<Tile> tile);
    virtual bool empty();
    virtual bool tile_queue_empty();
  protected:
    typedef struct {
      uint32_t id;
      uint32_t request_id;
      std::string name;
      bool launched;
      cycle_type start_cycle;
      cycle_type finish_cycle;
      cycle_type memory_stall_cycle;
      uint32_t total_tiles;
      uint32_t remain_tiles;
      uint32_t finished_tiles;
      uint32_t launched_tiles;
    } LayerStat;

    int _core_rr_id = 0;
    const cycle_type* _core_cycle;
    const uint64_t* _core_time;
    std::map<uint32_t, std::vector<uint32_t>> _partition_map;
    std::map<uint32_t, uint32_t> _cpu_to_partition;
    std::deque<Request> _request_queue;
    std::map<uint32_t, std::deque<std::unique_ptr<Tile>>> _executable_tile_queue;
    std::map<uint32_t, std::deque<std::unique_ptr<Tile>>> _core_executable_tile_queue;
    uint32_t _nr_layer = 0; // For layer round-robin
    SimulationConfig _config;
    robin_hood::unordered_map<uint32_t, LayerStat> _layer_stat_map;
    robin_hood::unordered_map<uint32_t, LayerStat> _active_layers_map;
    virtual void refresh_status();
    uint32_t count_active_layers();
    uint32_t cpu_to_partition(uint32_t cpu);
};

class TimeMultiplexScheduler : public Scheduler {
  public:
    TimeMultiplexScheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time);
    virtual void finish_tile(uint32_t core_id, std::unique_ptr<Tile> tile) override ;
  
  protected:
    virtual void refresh_status() override;
  private:
    uint32_t _request_rr=0;
};

class DedicatedCPUScheduler: public TimeMultiplexScheduler {
  public:
    DedicatedCPUScheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time);

  protected:
    virtual void refresh_status() override;
  private:
    uint32_t _request_rr=0;
};

class HalfSplitScheduler : public Scheduler {
  public:
    HalfSplitScheduler(SimulationConfig config, const cycle_type* core_cycle, const uint64_t* core_time);
    virtual void schedule_model(std::unique_ptr<Model> model, uint32_t sampe_size) override;
    virtual std::unique_ptr<Tile> get_tile(uint32_t core_id) override;
    virtual void finish_tile(uint32_t core_id, std::unique_ptr<Tile> tile) override ;
    
  protected:
    virtual void refresh_status() override;
    robin_hood::unordered_map<uint32_t, std::deque<std::unique_ptr<Tile>>> _executable_tile_queue_table;
};
