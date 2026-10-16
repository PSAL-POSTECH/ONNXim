#include "Mapping.h"
#include "SimulationConfig.h"
#include "gtest/gtest.h"

namespace {
/* 128x128 core of the TPUv4 configs (32 MB spad, 4 MB accum) */
SimulationConfig tpuv4_cores(uint32_t cores, uint32_t accum_spad_kb = 4096) {
  SimulationConfig config;
  config.num_cores = cores;
  config.core_config = new CoreConfig[cores];
  for (uint32_t i = 0; i < cores; i++) {
    config.core_config[i].core_type = CoreType::SYSTOLIC_WS;
    config.core_config[i].core_height = 128;
    config.core_config[i].core_width = 128;
    config.core_config[i].spad_size = 32768;
    config.core_config[i].accum_spad_size = accum_spad_kb;
  }
  config.precision = 2;
  config.dram_req_size = 32;
  return config;
}

void expect_covering_mapping(uint32_t cores, uint32_t n, uint32_t c, uint32_t m) {
  SCOPED_TRACE(std::to_string(cores) + " cores, N" + std::to_string(n) + " C" + std::to_string(c) +
               " M" + std::to_string(m));
  SimulationConfig config = tpuv4_cores(cores);
  MappingTable table(config);
  Mapping::LoopCounts key{.N = n, .C = c, .M = m, .S = 1, .R = 1, .Q = 1, .P = 1};
  const Mapping& mapping = table.fallback_mapping(key);
  const auto& in = mapping.tile_in_loop;
  const auto& out = mapping.tile_out_loop;
  /* At least one array per tile, and the tiles cover the whole GEMM */
  EXPECT_GE(in.N, 128u);
  EXPECT_GE(in.C, 128u);
  EXPECT_GE(in.M, 128u);
  EXPECT_GE(uint64_t(out.N) * in.N, n);
  EXPECT_GE(uint64_t(out.C) * in.C, c);
  EXPECT_GE(uint64_t(out.M) * in.M, m);
  delete[] config.core_config;
}
}  // namespace

/* A small GEMM split over many cores must not get a zero-sized inner tile */
TEST(GemmMappingTest, SmallGemmOnManyCores) {
  expect_covering_mapping(16, 1, 512, 1000);  // ResNet18 FC on 16 cores (#30)
  expect_covering_mapping(16, 1, 768, 768);
  expect_covering_mapping(4, 1, 768, 384);
  expect_covering_mapping(4, 512, 768, 768);  // these two already worked
  expect_covering_mapping(4, 1, 2048, 1000);
}

/* 64 KB accum can't hold a 128x128 tile: should exit with an error, not divide by zero */
TEST(GemmMappingDeathTest, AccumTooSmall) {
  SimulationConfig config = tpuv4_cores(1, 64);
  MappingTable table(config);
  Mapping::LoopCounts key{.N = 128, .C = 128, .M = 128, .S = 1, .R = 1, .Q = 1, .P = 1};
  EXPECT_EXIT(table.fallback_mapping(key), ::testing::ExitedWithCode(EXIT_FAILURE), "");
  delete[] config.core_config;
}
