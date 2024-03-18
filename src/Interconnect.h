#ifndef INTERCONNECT_H
#define INTERCONNECT_H
#include "Common.h"
#include "booksim2/Interconnect.hpp"
#include "helper/HelperFunctions.h"

class Interconnect {
 public:
  virtual ~Interconnect() = default;
  virtual bool running() = 0;
  virtual void cycle() = 0;
  virtual void push(uint32_t src, uint32_t dest, MemoryAccess* request) = 0;
  virtual bool is_full(uint32_t src, MemoryAccess* request) = 0;
  virtual bool is_empty(uint32_t nid) = 0;
  virtual MemoryAccess* top(uint32_t nid) = 0;
  virtual void pop(uint32_t nid) = 0;
  virtual void print_stats() = 0;

 protected:
  SimulationConfig _config;
  uint32_t _n_nodes;
  uint64_t _cycles;
};

// Simple without conflict interconnect
class SimpleInterconnect : public Interconnect {
 public:
  SimpleInterconnect(SimulationConfig config);
  virtual bool running() override;
  virtual void cycle() override;
  virtual void push(uint32_t src, uint32_t dest,
                    MemoryAccess* request) override;
  virtual bool is_full(uint32_t src, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t nid) override;
  virtual MemoryAccess* top(uint32_t nid) override;
  virtual void pop(uint32_t nid) override;
  virtual void print_stats() override {}

 private:
  uint32_t _latency;
  double _bandwidth;
  uint32_t _rr_start;
  uint32_t _buffer_size;

  struct Entity {
    cycle_type finish_cycle;
    uint32_t dest;
    MemoryAccess* access;
  };

  std::vector<std::queue<Entity>> _in_buffers;
  std::vector<std::queue<MemoryAccess*>> _out_buffers;
  std::vector<bool> _busy_node;
};

class Booksim2Interconnect : public Interconnect {
 public:
  Booksim2Interconnect(SimulationConfig config);
  virtual bool running() override;
  virtual void cycle() override;
  virtual void push(uint32_t src, uint32_t dest,
                    MemoryAccess* request) override;
  virtual bool is_full(uint32_t src, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t nid) override;
  virtual MemoryAccess* top(uint32_t nid) override;
  virtual void pop(uint32_t nid) override;
  virtual void print_stats() override;

 private:
  uint32_t _ctrl_size;
  std::string _config_path;
  std::unique_ptr<booksim2::Interconnect> _booksim;

  booksim2::Interconnect::Type get_booksim_type(MemoryAccess* access);
  uint32_t get_packet_size(MemoryAccess* access);
};
#endif