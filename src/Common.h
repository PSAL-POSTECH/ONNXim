#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "SimulationConfig.h"
#include "Stat.h"
#include "helper/HelperFunctions.h"
#include "nlohmann/json.hpp"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

#define SPAD_BASE 0x10000000
#define ACCUM_SPAD_BASE 0x20000000
#define GARBEGE_ADDR 0xFFFFFFFFFFFFFFF
#define KB *1024

#define PAGE_SIZE 4096

using json = nlohmann::json;

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

typedef struct {
  uint32_t id;
  addr_type dram_address;
  addr_type spad_address;
  uint64_t size;
  bool write;
  bool request;
  uint32_t core_id;
  cycle_type start_cycle;
  cycle_type dram_enter_cycle;
  cycle_type dram_finish_cycle;
  int buffer_id;
} MemoryAccess;

enum class Opcode {
  MOVIN,
  MOVOUT,
  MOVOUT_POOL,
  GEMM_PRELOAD,
  GEMM,
  GEMM_WRITE,
  COMP,
  IM2COL,
  SOFTMAX,
  LAYERNORM,
  ADD,
  GELU,
  BAR
};

typedef struct {
  Opcode opcode;
  cycle_type start_cycle;
  cycle_type finish_cycle;
  std::string id;
  std::vector<std::string> dependent_ids;
  std::string dest_id;
  addr_type dest_addr;
  uint32_t size;          // Used for sram allocation. Multiple of _config.dram_req_size
  uint32_t compute_size;
  std::vector<addr_type> src_addrs;
  int spad_id;
  int accum_spad_id;
  uint32_t operand_id  = 0;
  addr_type base_addr;

  uint32_t tile_m;
  uint32_t tile_k;
  uint32_t tile_n;

  bool src_from_accum = false;
  bool zero_init = false;
} Instruction;

typedef struct {
  enum class Status {
    INITIALIZED,
    RUNNING,
    FINISH,
    BAR,
    EMPTY,
  };
  Status status = Status::EMPTY;
  std::string optype;
  uint32_t layer_id;
  uint32_t fused_op_id; /* For fused operation */
  uint32_t batch;
  uint32_t Q;
  uint32_t P;
  uint32_t M;
  uint32_t C;
  uint32_t S;
  uint32_t R;

  TileStat stat;
  std::deque<std::unique_ptr<Instruction>> instructions;
  bool accum;
  bool skip;
  int spad_id;
  int accum_spad_id;
  int core_id = -1;
} Tile;

uint32_t generate_id();
uint32_t generate_mem_access_id();
addr_type allocate_address(uint32_t size);
SimulationConfig initialize_config(json config);