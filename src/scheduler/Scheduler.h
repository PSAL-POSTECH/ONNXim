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
    Scheduler(SimulationConfig config, const cycle_type* core_cycle);
    virtual void schedule_model(std::unique_ptr<Model> model, uint32_t sampe_size);
    virtual Tile get_tile(uint32_t core_id);
    virtual void finish_tile(uint32_t core_id, Tile tile);
    virtual bool empty();
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
    
    const cycle_type* _core_cycle;
    std::deque<Request> _request_queue;
    std::deque<Tile> _executable_tile_queue;
    SimulationConfig _config;
    robin_hood::unordered_map<uint32_t, LayerStat> _layer_stat_map;
    robin_hood::unordered_map<uint32_t, LayerStat> _active_layers_map;
    virtual void refresh_status();
    uint32_t count_active_layers();
};

class TimeMultiplexScheduler : public Scheduler {
  public:
    TimeMultiplexScheduler(SimulationConfig config, const cycle_type* core_cycle);
    // virtual void schedule_model(std::unique_ptr<Model> model, uint32_t sampe_size) override;
    virtual Tile get_tile(uint32_t core_id) override;
    virtual void finish_tile(uint32_t core_id, Tile tile) override ;
  
  protected:
    virtual void refresh_status() override;
  private:
    uint32_t _request_rr;
};

class HalfSplitScheduler : public Scheduler {
  public:
    HalfSplitScheduler(SimulationConfig config, const cycle_type* core_cycle);
    virtual void schedule_model(std::unique_ptr<Model> model, uint32_t sampe_size) override;
    virtual Tile get_tile(uint32_t core_id) override;
    virtual void finish_tile(uint32_t core_id, Tile tile) override ;
    
  protected:
    virtual void refresh_status() override;
    robin_hood::unordered_map<uint32_t, std::deque<Tile>> _executable_tile_queue_table;
};
