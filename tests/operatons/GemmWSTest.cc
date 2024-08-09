#include <filesystem>

#include "Common.h"
#include "Core.h"
#include "Model.h"
#include "SimulationConfig.h"
#include "SystolicWS.h"
#include "gtest/gtest.h"
#include "operations/GemmWS.h"
#include "operations/OperationFactory.h"
#define CYCLE_LIMIT 1e10

SimulationConfig get_default_config() {
  SimulationConfig config;
  config.num_cores = 1;
  config.core_type = CoreType::SYSTOLIC_WS;
  config.num_cores = 1;
  config.core_height = 8;
  config.core_width = 8;
  config.spad_size = 64;
  config.accum_spad_size = 16;
  config.precision = 1;
  config.dram_req_size = 32;
  config.layout = "NHWC";
  config.core_print_interval = 100000;
  return config;
}

GemmWS make_GemmWS(SimulationConfig config, std::string mapping_str, uint32_t n,
                   uint32_t c, uint32_t m) {
  std::string input_name = "input";
  std::vector<uint32_t> input_dims = {n, c};

  MappingTable mapping_table = MappingTable(config);
  Mapping mapping(mapping_str);

  Mapping::LoopCounts key{
      .N = n, .C = c, .M = m, .S = 1, .R = 1, .Q = 1, .P = 1};
  mapping_table[key] = mapping;
  std::vector<uint32_t> output_shape = {n, m};
  std::vector<uint32_t> weight_shape = {c, m};
  GemmWS op(config, mapping_table, input_dims, weight_shape, output_shape);
  op.initialize_tiles(mapping_table);
  return op;
}

void do_simulation(Core& core, Operation& op) {
  std::deque<std::unique_ptr<Tile>>& tiles = op.get_tiles();

  cycle_type cycle = 0;
  std::unique_ptr<Tile> running_tile;
  while (core.running() || !tiles.empty()) {
    if (core.can_issue() && !tiles.empty()) {
      running_tile = std::move(tiles.front());
      tiles.pop_front();
      core.issue(std::move(running_tile));
    }
    if (core.has_memory_request()) {
      /* Assume Magic memory */
      MemoryAccess* access = core.top_memory_request();
      access->request = false;
      core.pop_memory_request();
      core.push_memory_response(access);
    }
    core.cycle();
    cycle++;
    /* Kill simulation if infinity loop */
    if (cycle > CYCLE_LIMIT) break;
  }
  core.print_stats();
}

/* ResNet18 FC Layer */
TEST(GemmWS1x512x1000Test, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C512 M1000 - [O] N1 C2 M16 - [I] N1 C448 M64";
  /* Input information */
  uint32_t n = 1, c = 512, m = 1000;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 66360;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

/* ResNet50 FC Layer */

TEST(GemmWS1x2048x1000Test, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C2048 M1000 - [O] N1 C5 M16 - [I] N1 C448 M64";
  /* Input information */
  uint32_t n = 1, c = 2048, m = 1000;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 270431;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}