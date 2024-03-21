#pragma once

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

enum class CoreType { SYSTOLIC_OS, SYSTOLIC_WS };

enum class DramType { SIMPLE, RAMULATOR };

enum class IcntType { SIMPLE, BOOKSIM2 };

struct SimulationConfig {
  /* Core config */
  uint32_t num_cores;
  CoreType core_type;
  uint32_t core_freq;
  uint32_t core_width;
  uint32_t core_height;

  /* Vector config*/
  uint32_t vector_process_bit;
  uint32_t layernorm_latency;
  uint32_t softmax_latency;
  uint32_t add_latency;
  uint32_t mul_latency;
  uint32_t exp_latency;
  uint32_t gelu_latency;
  uint32_t add_tree_latency;
  uint32_t scalar_sqrt_latency;
  uint32_t scalar_add_latency;
  uint32_t scalar_mul_latency;

  /* SRAM config */
  uint32_t sram_width;
  uint32_t spad_size;
  uint32_t accum_spad_size;

  /* DRAM config */
  DramType dram_type;
  uint32_t dram_freq;
  uint32_t dram_channels;
  uint32_t dram_req_size;
  uint32_t dram_latency;
  std::string dram_config_path;

  /* ICNT config */
  IcntType icnt_type;
  std::string icnt_config_path;
  uint32_t icnt_freq;
  uint32_t icnt_latency;

  /* Sheduler config */
  std::string scheduler_type;

  /* Other configs */
  uint32_t precision;
  std::string layout;

  uint64_t align_address(uint64_t addr) {
    return addr - (addr % dram_req_size);
  }
};