
#include "Common.h"
#include "Core.h"
#include "SimulationConfig.h"
#include "SystolicWS.h"
#include "gtest/gtest.h"
#include "operations/ConvWS.h"

TEST(SystolicWSTileExecutionTest, BasicAssertions) {
  /* Weight statinary config*/
  SimulationConfig config;
  config.num_cores = 1;
  config.core_config = new struct CoreConfig;
  config.core_config[0].core_type = CoreType::SYSTOLIC_WS;
  config.core_config[0].core_height = 8;
  config.core_config[0].core_width = 8;
  config.core_config[0].spad_size = 192;
  config.core_config[0].accum_spad_size = 192;
  config.precision = 4;
  config.dram_req_size = 32;
  
  SystolicWS core(0, config);
  std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
            .status = Tile::Status::INITIALIZED,
            .layer_id = 0,
            .spad_id = 0,
            .accum_spad_id = 0});
          
  tile->instructions.push_back(std::make_unique<Instruction>(
      Instruction{.opcode = Opcode::GEMM_PRELOAD,
                  .dest_addr = ACCUM_SPAD_BASE,
                  .compute_size = 8,
                  .src_addrs = std::vector<addr_type>{}}));

  core.issue(std::move(tile));
  cycle_type cycle = 0;
  while (core.running()) {
    core.cycle();
    if (core.has_memory_request()) {
      MemoryAccess* access = core.top_memory_request();
      access->request = false;
      core.pop_memory_request();
      core.push_memory_response(access);
    }
    cycle++;
    if (cycle > 1000) break;
  }
  /* Weight load 7 + Preload 8 + Single mul 8 + Mesh execution 15 + Output delay 1 = 39 cycles*/
  ASSERT_EQ(cycle, 39);
}

TEST(SystolicWSTwoGemmExecutionTest, BasicAssertions) {
  /* Weight statinary config*/
  SimulationConfig config;
  config.num_cores = 1;
  config.core_config = new struct CoreConfig;
  config.core_config[0].core_type = CoreType::SYSTOLIC_WS;
  config.core_config[0].core_height = 8;
  config.core_config[0].core_width = 8;
  config.core_config[0].spad_size = 192;
  config.core_config[0].accum_spad_size = 192;
  config.precision = 4;
  config.dram_req_size = 32;

  SystolicWS core(0, config);
  std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
            .status = Tile::Status::INITIALIZED,
            .layer_id = 0,
            .spad_id = 0,
            .accum_spad_id = 0});

  tile->instructions.push_back(std::make_unique<Instruction>(
      Instruction{.opcode = Opcode::GEMM_PRELOAD,
                  .dest_addr = ACCUM_SPAD_BASE,
                  .compute_size = 8,
                  .src_addrs = std::vector<addr_type>{}}));
  tile->instructions.push_back(std::make_unique<Instruction>(
      Instruction{.opcode = Opcode::GEMM_PRELOAD,
                  .dest_addr = ACCUM_SPAD_BASE,
                  .compute_size = 8,
                  .src_addrs = std::vector<addr_type>{}}));

  core.issue(std::move(tile));
  cycle_type cycle = 0;
  while (core.running()) {
    core.cycle();
    if (core.has_memory_request()) {
      MemoryAccess* access = core.top_memory_request();
      access->request = false;
      core.pop_memory_request();
      core.push_memory_response(access);
    }
    cycle++;
    if (cycle > 1000) break;
  }
  /* Weight load 7 + Preload 8 + Single mul 8 + Mesh execution 23 + Output delay 1= 47 cycles*/
  ASSERT_EQ(cycle, 47);
}