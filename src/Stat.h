#pragma once

#include <stdint.h>

#include <vector>

typedef struct {
  uint64_t start_cycle;
  uint64_t cycles;
  uint64_t compute_cycles;
  uint64_t memory_stall;
  uint64_t dependency_stall;
  uint64_t sram_reads;
  uint64_t sram_writes;
} TileStat;

typedef struct {
  uint64_t op_cycles;
  std::vector<TileStat> tile_stats;
} OpStat;

typedef struct {
  uint64_t total_cycles;
  std::vector<OpStat> op_stats;
} ModelStat;
