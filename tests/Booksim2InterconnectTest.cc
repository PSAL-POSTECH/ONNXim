#include <cstdlib>
#include <filesystem>
#include <set>
#include <vector>

#include "Interconnect.h"
#include "SimulationConfig.h"
#include "gtest/gtest.h"

/* 4 cores x 16 ports + 8 channels = 72 nodes (fly_c64_m8). Every core port sends a read to a
   channel and gets the reply back. Node ids as in Simulator::get_dest_node. */
TEST(Booksim2InterconnectTest, InjectionPorts) {
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path();
  setenv("ONNXIM_HOME", root.c_str(), 1);
  SimulationConfig config;
  config.num_cores = 4;
  config.icnt_injection_ports_per_core = 16;
  config.dram_channels = 8;
  config.dram_req_size = 32;
  config.icnt_config_path = "../configs/booksim2_configs/fly_c64_m8.icnt";
  Booksim2Interconnect network(config);

  const uint32_t ports = config.num_cores * config.icnt_injection_ports_per_core;
  std::vector<MemoryAccess> requests(ports);
  for (uint32_t port = 0; port < ports; ++port) {
    requests[port] = MemoryAccess{};
    requests[port].id = port;
    requests[port].size = config.dram_req_size;
    requests[port].request = true;
    requests[port].write = false;
  }
  std::vector<bool> sent(ports, false);
  std::set<uint32_t> replies;
  for (uint32_t cycle = 0; cycle < 100000 && replies.size() < ports; ++cycle) {
    for (uint32_t port = 0; port < ports; ++port) {
      if (!sent[port] && !network.is_full(port, &requests[port])) {
        network.push(port, ports + port % config.dram_channels, &requests[port]);
        sent[port] = true;
      }
    }
    for (uint32_t ch = 0; ch < config.dram_channels; ++ch) {
      const uint32_t node = ports + ch;
      if (network.is_empty(node)) continue;
      MemoryAccess* request = network.top(node);
      EXPECT_EQ(request->id % config.dram_channels, ch);
      request->request = false;  // reply right away, no DRAM here
      if (!network.is_full(node, request)) {
        network.push(node, request->id, request);
        network.pop(node);
      }
    }
    for (uint32_t port = 0; port < ports; ++port) {
      if (network.is_empty(port)) continue;
      MemoryAccess* reply = network.top(port);
      EXPECT_EQ(reply->id, port);
      EXPECT_FALSE(reply->request);
      EXPECT_TRUE(replies.insert(reply->id).second);
      network.pop(port);
    }
    network.cycle();
  }
  EXPECT_EQ(replies.size(), ports);
}
