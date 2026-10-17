#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "Model.h"
#include "Tensor.h"
#include "gtest/gtest.h"
#include "operations/Attention.h"

// A long prefill must store every query row when queries span multiple tiles.
TEST(AttentionOutputRowsTest, EveryQueryRowIsStored) {
  SimulationConfig config;
  auto cores = std::make_unique<CoreConfig[]>(1);
  config.num_cores = 1;
  config.core_config = cores.get();
  config.core_config[0].core_type = CoreType::SYSTOLIC_WS;
  config.core_config[0].core_height = config.core_config[0].core_width = 128;
  config.core_config[0].spad_size = 32768;
  config.core_config[0].accum_spad_size = 4096;
  config.precision = 2;
  config.dram_req_size = 32;
  config.full_precision = 4;

  Model model(json::object(), config, "attention-output-rows");
  const uint32_t tokens = 2048, hidden = 4096, kv_dim = 1024;
  auto add_tensor = [&](const char* name, std::vector<uint32_t> dims) {
    auto tensor = std::make_unique<Tensor>(0, name, dims, config.precision, true);
    const uint32_t id = tensor->get_id();
    model.add_tensor(std::move(tensor));
    return id;
  };
  const uint32_t q = add_tensor("q", {tokens, hidden});
  const uint32_t k = add_tensor("k", {tokens, kv_dim});
  const uint32_t v = add_tensor("v", {tokens, kv_dim});
  std::map<std::string, std::string> attributes = {
      {"num_tokens", "2048"}, {"num_heads", "32"}, {"num_kv_heads", "8"}, {"hidden_size", "4096"}};
  Attention op(config, &model, "attention", attributes, 0);
  op.add_input(q);
  op.add_input(k);
  op.add_input(v);
  MappingTable mapping_table(config);
  op.initialize_tiles(mapping_table);

  Tensor* output = op.get_output(0);
  const uint64_t row_bytes = uint64_t(hidden) * config.precision;
  std::set<uint64_t> stored_rows;
  for (const auto& tile : op.get_tiles())
    for (const auto& inst : tile->instructions)
      if (inst->opcode == Opcode::MOVOUT)
        for (addr_type address : inst->src_addrs) {
          ASSERT_GE(address, output->get_address());
          ASSERT_LT(address, output->get_address() + output->get_size());
          stored_rows.insert((address - output->get_address()) / row_bytes);
        }
  EXPECT_EQ(stored_rows.size(), tokens);
}
