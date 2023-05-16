#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

enum class Opcode { MOVIN, MOVOUT, GEMM_PRELOAD, GEMM, GEMM_WRITE, COMP, BAR };

#define SPAD_BASE 0x10000000
#define ASPAD_BASE 0x20000000
typedef uint64_t addr_type;
typedef uint64_t cycle_type;

class Instruction {
 public:
  Instruction();
  std::string toString();
  
 private:
  enum class Type {
    LD_INST, ST_INST, EXE_INST
  };
  uint32_t id; 
  Opcode opcode;
  Type type;
  size_t tile_size;
  cycle_type start_cycle;
  cycle_type finish_cycle;
  std::vector<std::string> dependent_ids;
  std::string dest_id;
  addr_type spad_addr;
  uint32_t spad_size;
  std::vector<addr_type> dram_addrs;
};