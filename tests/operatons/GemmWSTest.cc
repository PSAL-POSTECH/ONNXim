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
  config.core_config = new struct CoreConfig;
  config.core_config[0].core_type = CoreType::SYSTOLIC_WS;
  config.core_config[0].core_height = 8;
  config.core_config[0].core_width = 8;
  config.core_config[0].spad_size = 64;
  config.core_config[0].accum_spad_size = 16;
  config.precision = 1;
  config.dram_req_size = 32;
  config.layout = "NHWC";
  config.core_print_interval = 10000000;
  return config;
}

GemmWS make_GemmWS(SimulationConfig config, std::string mapping_str, uint32_t n,
                   uint32_t c, uint32_t m) {
  std::string input_name = "input";
  std::vector<uint32_t> input_dims = {1, n, c};

  MappingTable mapping_table = MappingTable(config);
  Mapping mapping(mapping_str);

  Mapping::LoopCounts key{
      .N = n, .C = c, .M = m, .S = 1, .R = 1, .Q = 1, .P = 1};
  mapping_table[key] = mapping;
  std::vector<uint32_t> output_shape = {1, n, m};
  std::vector<uint32_t> weight_shape = {c, m};
  GemmWS op(config, mapping_table, input_dims, weight_shape, output_shape, 0);
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
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
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
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(Bert_512x512x1024, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C512 M1024 - [O] N13 C2 M26 - [I] N40 C256 M40";
  /* Input information */
  uint32_t n = 512, c = 512, m = 1024;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 4227300;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(Bert_512x1024x2, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C1024 M2 - [O] N8 C3 M1 - [I] N64 C448 M2";
  /* Input information */
  uint32_t n = 512, c = 1024, m = 2;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 67553;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(Bert_512x1024x512, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C1024 M512 - [O] N13 C3 M13 - [I] N40 C342 M40";
  /* Input information */
  uint32_t n = 512, c = 1024, m = 512;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 4228207;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(Bert_512x1024x1024, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C1024 M1024 - [O] N13 C3 M26 - [I] N40 C342 M40";
  /* Input information */
  uint32_t n = 512, c = 1024, m = 1024;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 8458619;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(Bert_512x1024x3072, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C1024 M3072 - [O] N13 C3 M77 - [I] N40 C342 M40";
  /* Input information */
  uint32_t n = 512, c = 1024, m = 3072;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 25390266;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(Bert_512x1024x4096, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C1024 M4096 - [O] N13 C3 M103 - [I] N40 C342 M40";
  /* Input information */
  uint32_t n = 512, c = 1024, m = 4096;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 33912072;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(Bert_512x4096x1024, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C4096 M1024 - [O] N13 C11 M26 - [I] N40 C373 M40";
  /* Input information */
  uint32_t n = 512, c = 4096, m = 1024;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 33912533;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(GPT_512x512x768, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C512 M768 - [O] N13 C2 M20 - [I] N40 C256 M40";
  /* Input information */
  uint32_t n = 512, c = 512, m = 768;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 3171978;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(GPT_512x768x512, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C768 M512 - [O] N13 C2 M13 - [I] N40 C384 M40";
  /* Input information */
  uint32_t n = 512, c = 768, m = 512;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 3170315;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(GPT_512x768x768, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C768 M768 - [O] N13 C2 M20 - [I] N40 C384 M40";
  /* Input information */
  uint32_t n = 512, c = 768, m = 768;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 4756894;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}


TEST(GPT_512x768x2304, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N512 C768 M2304 - [O] N13 C2 M58 - [I] N40 C384 M40";
  /* Input information */
  uint32_t n = 512, c = 768, m = 2304;

  /* Weight statinary config*/
  SimulationConfig config = get_default_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  GemmWS op = make_GemmWS(config, test_mapping, n, c, m);

  do_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 14266861;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}