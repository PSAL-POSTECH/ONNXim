#include "Common.h"

uint32_t generate_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}
uint32_t generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}

addr_type allocate_address(uint32_t size) {
  static addr_type base_addr{0};
  addr_type result = base_addr;
  int offset = 0;
  if (result % 256 != 0) {
    offset = 256 - (result % 256);
  }
  result += offset;
  assert(result % 256 == 0);
  base_addr += (size + offset);
  base_addr += (256 - base_addr % 256);
  return result;
}

SimulationConfig initialize_config(json config) {
  SimulationConfig parsed_config;

  /* Core configs */
  parsed_config.num_cores = config["num_cores"];
  if ((std::string)config["core_type"] == "systolic_os")
    parsed_config.core_type = CoreType::SYSTOLIC_OS;
  else if ((std::string)config["core_type"] == "systolic_ws")
    parsed_config.core_type = CoreType::SYSTOLIC_WS;
  else
    throw std::runtime_error(fmt::format("Not implemented core type {} ",
                                         (std::string)config["core_type"]));
  parsed_config.core_freq = config["core_freq"];
  parsed_config.core_width = config["core_width"];
  parsed_config.core_height = config["core_height"];

  /* Vector configs */
  parsed_config.vector_process_bit = config["vector_process_bit"];
  parsed_config.add_latency = config["add_latency"];
  parsed_config.mul_latency = config["mul_latency"];
  parsed_config.exp_latency = config["exp_latency"];
  parsed_config.gelu_latency = config["gelu_latency"];
  parsed_config.add_tree_latency = config["add_tree_latency"];
  parsed_config.scalar_sqrt_latency = config["scalar_sqrt_latency"];
  parsed_config.scalar_add_latency = config["scalar_add_latency"];
  parsed_config.scalar_mul_latency = config["scalar_mul_latency"];

  /* SRAM configs */
  parsed_config.sram_width = config["sram_width"];
  parsed_config.spad_size = config["spad_size"];
  parsed_config.accum_spad_size = config["accum_spad_size"];

  /* DRAM config */
  if ((std::string)config["dram_type"] == "simple")
    parsed_config.dram_type = DramType::SIMPLE;
  else if ((std::string)config["dram_type"] == "ramulator")
    parsed_config.dram_type = DramType::RAMULATOR;
  else
    throw std::runtime_error(fmt::format("Not implemented dram type {} ",
                                         (std::string)config["dram_type"]));
  parsed_config.dram_freq = config["dram_freq"];
  if (config.contains("dram_latency"))
    parsed_config.dram_latency = config["dram_latency"];
  if (config.contains("dram_config_path"))
    parsed_config.dram_config_path = config["dram_config_path"];
  parsed_config.dram_channels = config["dram_channels"];
  if (config.contains("dram_req_size"))
    parsed_config.dram_req_size = config["dram_req_size"];
  if (config.contains("dram_print_interval"))
    parsed_config.dram_print_interval = config["dram_print_interval"];

  /* Icnt config */
  if ((std::string)config["icnt_type"] == "simple")
    parsed_config.icnt_type = IcntType::SIMPLE;
  else if ((std::string)config["icnt_type"] == "booksim2")
    parsed_config.icnt_type = IcntType::BOOKSIM2;
  else
    throw std::runtime_error(fmt::format("Not implemented icnt type {} ",
                                         (std::string)config["icnt_type"]));
  parsed_config.icnt_freq = config["icnt_freq"];
  if (config.contains("icnt_latency"))
    parsed_config.icnt_latency = config["icnt_latency"];
  if (config.contains("icnt_config_path"))
    parsed_config.icnt_config_path = config["icnt_config_path"];

  parsed_config.scheduler_type = config["scheduler"];
  parsed_config.precision = config["precision"];
  parsed_config.layout = config["layout"];

  if (config.contains("partition")) {
    for (int i=0; i<parsed_config.num_cores; i++) {
      std::string core_partition = "core_" + std::to_string(i);
      uint32_t partition_id = uint32_t(config["partition"][core_partition]);
      parsed_config.partiton_map[partition_id].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, partition_id);
    }
  } else {
    /* Default: all partition 0 */
    for (int i=0; i<parsed_config.num_cores; i++) {
      parsed_config.partiton_map[0].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, 0);
    }
  }
  return parsed_config;
}

uint32_t ceil_div(uint32_t src, uint32_t div) { return (src+div-1)/div; }

std::vector<uint32_t> parse_dims(const std::string &str) {
  std::vector<uint32_t> dims;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, ',')) {
      dims.push_back(std::stoi(token));
  }
  return dims;
}

std::string dims_to_string(const std::vector<uint32_t> &dims){
  std::string str;
  for (int i=0; i<dims.size(); i++) {
    str += std::to_string(dims[i]);
    if (i != dims.size()-1) {
      str += ",";
    }
  }
  return str;
}