#include <filesystem>

#include "Common.h"
#include "Core.h"
#include "Model.h"
#include "SimulationConfig.h"
#include "SystolicWS.h"
#include "gtest/gtest.h"
#include "operations/ConvWS.h"
#include "operations/OperationFactory.h"
#define CYCLE_LIMIT 1e10

namespace fs = std::filesystem;

SimulationConfig get_default_conv_config() {
  SimulationConfig config;
  config.core_type = CoreType::SYSTOLIC_WS;
  config.num_cores = 1;
  config.core_height = 8;
  config.core_width = 8;
  config.spad_size = 1024;
  config.accum_spad_size = 1024;
  config.precision = 4;
  config.dram_req_size = 32;
  config.layout = "NHWC";
  config.core_print_interval = 100000;
  return config;
}

ConvWS make_ConvWS(SimulationConfig config, std::string mapping_str, convInfo info) {
  std::string input_name = "input";
  MappingTable mapping_table = MappingTable(config);
  Mapping mapping(mapping_str);

  Mapping::LoopCounts key{
      .N = info.conv_out_shape[0],
      .C = info.weight_shape[1],
      .M = info.weight_shape[0],
      .S = info.weight_shape[2],
      .R = info.weight_shape[3],
      .Q = info.conv_out_shape[1],
      .P = info.conv_out_shape[2]};
  mapping_table[key] = mapping;
  ConvWS op(config, mapping_table, info);
  op.initialize_tiles(mapping_table);
  return op;
}

void do_conv_simulation(Core& core, Operation& op) {
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

/*ResNet 1st layer*/
TEST(ConvWSTestResNetConv_0, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C3 M64 P112 Q112 S7 R7 - [O] N1 C1 M4 P5 Q6 S1 R1 - [I] N1 C3 M16 P23 Q22 S7 R7";
  /* Input information */
  convInfo info = {
    .kernel_shape = {7, 7},
    .strides = {2, 2},
    .dilations = {1, 1},
    .pads = {3, 3, 3, 3},
    .input_shape = {1, 224, 224, 3},
    .weight_shape = {64, 3, 7, 7},
    .conv_out_shape = {1, 112, 112, 64},
    .group = 1,
    .activation_fused = true,
    .activation_type = "Relu",
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 4938719;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetConv_3_8, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C64 M64 P56 Q56 S3 R3 - [O] N1 C1 M4 P3 Q3 S1 R1 - [I] N1 C64 M16 P23 Q22 S3 R3";
  /* Input information */
  convInfo info = {
    .kernel_shape = {3, 3},
    .strides = {1, 1},
    .dilations = {1, 1},
    .pads = {1, 1, 1, 1},
    .input_shape = {1, 56, 56, 64},
    .weight_shape = {64, 64, 3, 3},
    .conv_out_shape = {1, 56, 56, 64},
    .group = 1,
    .activation_fused = true,
    .activation_type = "Relu",
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 1881975;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetConv_13, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C64 M128 P28 Q28 S3 R3 - [O] N1 C2 M8 P2 Q2 S1 R1 - [I] N1 C51 M16 P23 Q22 S3 R3";
  /* Input information */
  convInfo info = {
    .kernel_shape = {3, 3},
    .strides = {2, 2},
    .dilations = {1, 1},
    .pads = {1, 1, 1, 1},
    .input_shape = {1, 56, 56, 64},
    .weight_shape = {128, 64, 3, 3},
    .conv_out_shape = {1, 28, 28, 128},
    .group = 1,
    .activation_fused = true,
    .activation_type = "Relu",
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 1027536;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetConv_16, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C64 M128 P28 Q28 S1 R1 - [O] N1 C1 M8 P2 Q2 S1 R1 - [I] N1 C64 M16 P23 Q22 S1 R1";
  /* Input information */
  convInfo info = {
    .kernel_shape = {1, 1},
    .strides = {2, 2},
    .dilations = {1, 1},
    .pads = {0, 0, 0, 0},
    .input_shape = {1, 56, 56, 64},
    .weight_shape = {128, 64, 1, 1},
    .conv_out_shape = {1, 28, 28, 128},
    .group = 1,
    .activation_fused = false,
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 101941;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetConv_15_19, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C128 M128 P28 Q28 S3 R3 - [O] N1 C1 M8 P2 Q2 S1 R1 - [I] N1 C128 M16 P23 Q22 S3 R3";
  /* Input information */
  convInfo info = {
    .kernel_shape = {3, 3},
    .strides = {1, 1},
    .dilations = {1, 1},
    .pads = {1, 1, 1, 1},
    .input_shape = {1, 28, 28, 128},
    .weight_shape = {128, 128, 3, 3},
    .conv_out_shape = {1, 28, 28, 128},
    .group = 1,
    .activation_fused = true,
    .activation_type = "Relu",
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 1825897;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetConv_24, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C128 M256 P14 Q14 S3 R3 - [O] N1 C2 M7 P1 Q1 S1 R1 - [I] N1 C104 M40 P14 Q14 S3 R3";
  /* Input information */
  convInfo info = {
    .kernel_shape = {3, 3},
    .strides = {2, 2},
    .dilations = {1, 1},
    .pads = {1, 1, 1, 1},
    .input_shape = {1, 28, 28, 128},
    .weight_shape = {256, 128, 3, 3},
    .conv_out_shape = {1, 14, 14, 256},
    .group = 1,
    .activation_fused = true,
    .activation_type = "Relu",
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 912752;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSAddrTest, BasicAssertions) {}

TEST(ConvWSTestResNetConv_26, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C256 M256 P14 Q14 S3 R3 - [O] N1 C2 M7 P1 Q1 S1 R1 - [I] N1 C210 M40 P14 Q14 S3 R3";
  /* Input information */
  convInfo info = {
    .kernel_shape = {1, 1},
    .strides = {1, 1},
    .dilations = {1, 1},
    .pads = {1, 1, 1, 1},
    .input_shape = {1, 14, 14, 256},
    .weight_shape = {256, 256, 1, 1},
    .conv_out_shape = {1, 14, 14, 256},
    .group = 1,
    .activation_fused = false,
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 1882561;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetConv_27, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C128 M256 P14 Q14 S1 R1 - [O] N1 C1 M7 P1 Q1 S1 R1 - [I] N1 C128 M40 P14 Q14 S1 R1";
  /* Input information */
  convInfo info = {
    .kernel_shape = {1, 1},
    .strides = {2, 2},
    .dilations = {1, 1},
    .pads = {0, 0, 0, 0},
    .input_shape = {1, 28, 28, 128},
    .weight_shape = {256, 128, 3, 3},
    .conv_out_shape = {1, 14, 14, 256},
    .group = 1,
    .activation_fused = false,
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 101435;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetConv_35, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C256 M512 P7 Q7 S3 R3 - [O] N1 C3 M5 P1 Q1 S1 R1 - [I] N1 C109 M104 P7 Q7 S3 R3";
  /* Input information */
  convInfo info = {
    .kernel_shape = {3, 3},
    .strides = {2, 2},
    .dilations = {1, 1},
    .pads = {1, 1, 1, 1},
    .input_shape = {1, 14, 14, 256},
    .weight_shape = {512, 256, 3, 3},
    .conv_out_shape = {1, 7, 7, 512},
    .group = 1,
    .activation_fused = false,
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 951837;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetConv_38, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C256 M512 P7 Q7 S1 R1 - [O] N1 C1 M4 P1 Q1 S1 R1 - [I] N1 C256 M160 P7 Q7 S1 R1";
  /* Input information */
  convInfo info = {
    .kernel_shape = {1, 1},
    .strides = {2, 2},
    .dilations = {1, 1},
    .pads = {0, 0, 0, 0},
    .input_shape = {1, 14, 14, 256},
    .weight_shape = {512, 256, 1, 1},
    .conv_out_shape = {1, 7, 7, 512},
    .group = 1,
    .activation_fused = false,
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 102566;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}

TEST(ConvWSTestResNetLast, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C512 M512 P7 Q7 S3 R3 - [O] N1 C5 M5 P1 Q1 S1 R1 - [I] N1 C120 M112 P7 Q7 S3 R3";
  /* Input information */
  convInfo info = {
    .kernel_shape = {3, 3},
    .strides = {1, 1},
    .dilations = {1, 1},
    .pads = {1, 1, 1, 1},
    .input_shape = {1, 7, 7, 512},
    .weight_shape = {512, 512, 3, 3},
    .conv_out_shape = {1, 7, 7, 512},
    .group = 1,
    .activation_fused = false,
    .bathnorm_fused = false,
    .skip_connection_fused = false,
    .pool_fused = false
  };
  /* Weight statinary config*/
  SimulationConfig config = get_default_conv_config();
  OperationFactory::initialize(config);
  SystolicWS core(0, config);

  ConvWS op = make_ConvWS(config, test_mapping, info);

  do_conv_simulation(core, op);

  cycle_type compute_cycle = core.get_compute_cycles();
  cycle_type GT = 1845838;
  cycle_type diff = llabs(GT - compute_cycle);
  printf("Error Rate: %.2f %%\n", float(diff) / GT * 100.0);
  ASSERT_EQ(compute_cycle, GT);
}
