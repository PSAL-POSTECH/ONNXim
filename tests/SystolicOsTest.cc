
#include "Common.h"
#include "Core.h"
#include "SimulationConfig.h"
#include "SystolicOS.h"
#include "gtest/gtest.h"
#include "operations/ConvOS.h"

TEST(SystolicOSTileExecutionTest, BasicAssertions) {
  // /* Weight statinary config*/
  // SimulationConfig config;
  // config.core_type = CoreType::SYSTOLIC_OS;
  // config.core_height = 8;
  // config.core_width = 8;
  // config.spad_size = 192;
  // config.precision = 4;
  // config.dram_req_size = 32;

  // SystolicOS core(0, config);
  // Tile tile{.status = Tile::Status::INITIALIZED, .layer_id = 0};
  // tile.instructions.push(
  //     Instruction{.opcode = Opcode::MOVIN,
  //                 .id = "WEIGHT-0",
  //                 .addrs = std::vector<addr_type>{0x00, 0x20}});
  // tile.instructions.push(
  //     Instruction{.opcode = Opcode::GEMM,
  //                 .tile_size = 100,
  //                 .dependent_ids = std::vector<std::string>{"WEIGHT-0"}});
  // core.issue(&tile);
  // cycle_type cycle = 0;
  // while (tile.status != Tile::Status::FINISH) {
  //   core.cycle();
  //   if (core.has_memory_request()) {
  //     MemoryAccess* access = core.top_memory_request();
  //     access->request = false;
  //     core.pop_memory_request();
  //     core.push_memory_response(access);
  //   }
  //   cycle++;
  //   if (cycle > 1000) break;
  // }
  // /*TODO: insert cycle count from GEMMINI */
  // ASSERT_EQ(cycle, 125);
}