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
  config.num_cores = 1;
  config.core_config = new struct CoreConfig;
  config.core_config[0].core_type = CoreType::SYSTOLIC_WS;
  config.core_config[0].core_height = 8;
  config.core_config[0].core_width = 8;
  config.core_config[0].spad_size = 64+16+16;
  config.core_config[0].accum_spad_size = 16;
  config.precision = 1;
  config.dram_req_size = 32;
  config.layout = "NHWC";
  config.core_print_interval = 1000000;
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
TEST(ResNet18_Conv1, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C3 M64 P112 Q112 S7 R7 - [O] N1 C1 M8 P8 Q7 S1 R1 - [I] N1 C3 M8 P15 Q17 S7 R7";
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
  cycle_type GT = 5887961;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer1_0_Conv1, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C64 M64 P56 Q56 S3 R3 - [O] N1 C1 M8 P4 Q4 S1 R1 - [I] N1 C64 M8 P15 Q17 S3 R3";
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
  cycle_type GT = 2133641;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer2_0_Conv1, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C64 M128 P28 Q28 S3 R3 - [O] N1 C3 M16 P2 Q2 S1 R1 - [I] N1 C24 M8 P15 Q17 S3 R3";
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
  cycle_type GT = 1052613;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer2_0_downsample_0, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C64 M128 P28 Q28 S1 R1 - [O] N1 C1 M16 P2 Q2 S1 R1 - [I] N1 C64 M8 P15 Q17 S1 R1";
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
  cycle_type GT = 118304;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer2_0_Conv2, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C128 M128 P28 Q28 S3 R3 - [O] N1 C2 M16 P2 Q2 S1 R1 - [I] N1 C80 M8 P17 Q14 S3 R3";
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
  cycle_type GT = 2103568;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer3_0_Conv1, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C128 M256 P14 Q14 S3 R3 - [O] N1 C4 M32 P1 Q1 S1 R1 - [I] N1 C32 M8 P14 Q14 S3 R3";
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
  cycle_type GT = 914287;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer3_0_Conv2, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C256 M256 P14 Q14 S3 R3 - [O] N1 C3 M32 P1 Q1 S1 R1 - [I] N1 C96 M8 P14 Q14 S3 R3";
  /* Input information */
  convInfo info = {
    .kernel_shape = {3, 3},
    .strides = {1, 1},
    .dilations = {1, 1},
    .pads = {1, 1, 1, 1},
    .input_shape = {1, 14, 14, 256},
    .weight_shape = {256, 256, 3, 3},
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
  cycle_type GT = 1828936;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer3_0_downsample_0, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C128 M256 P14 Q14 S1 R1 - [O] N1 C1 M32 P1 Q1 S1 R1 - [I] N1 C128 M8 P14 Q14 S1 R1";
  /* Input information */
  convInfo info = {
    .kernel_shape = {1, 1},
    .strides = {2, 2},
    .dilations = {1, 1},
    .pads = {0, 0, 0, 0},
    .input_shape = {1, 28, 28, 128},
    .weight_shape = {256, 128, 1, 1},
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
  cycle_type GT = 101763;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer4_0_Conv1, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C256 M512 P7 Q7 S3 R3 - [O] N1 C6 M13 P1 Q1 S1 R1 - [I] N1 C51 M40 P7 Q7 S3 R3";
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
  cycle_type GT = 1040621;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer4_0_downsample_0, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C256 M512 P7 Q7 S1 R1 - [O] N1 C1 M13 P1 Q1 S1 R1 - [I] N1 C256 M40 P7 Q7 S1 R1";
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
  cycle_type GT = 102703;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}

TEST(ResNet18_layer4_0_Conv2, BasicAssertions) {
  /* User defined WS mapping information */
  std::string test_mapping = "[T] N1 C512 M512 P7 Q7 S3 R3 - [O] N1 C8 M13 P1 Q1 S1 R1 - [I] N1 C73 M40 P7 Q7 S3 R3";
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
  cycle_type GT = 2051216;
  cycle_type diff = llabs(GT - compute_cycle);
  float err = float(diff) / GT * 100.0;
  printf("Error Rate: %.2f %%, ONNXim: %ld, Gemmini: %ld\n", err, compute_cycle, GT);
  EXPECT_LT(err, 5.0);
}
