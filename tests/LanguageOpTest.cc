#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Model.h"
#include "Tensor.h"
#include "gtest/gtest.h"
#include "operations/BiasAct.h"
#include "operations/SkipLayerNorm.h"

namespace {
SimulationConfig language_op_config() {
  SimulationConfig config;
  config.num_cores = 1;
  config.core_config = new CoreConfig[1];
  config.core_config[0].core_type = CoreType::SYSTOLIC_WS;
  config.core_config[0].core_height = 8;
  config.core_config[0].core_width = 8;
  config.core_config[0].spad_size = 64;
  config.core_config[0].accum_spad_size = 16;
  config.precision = 2;
  config.dram_req_size = 32;
  return config;
}

uint32_t add_tensor(Model& model, const std::string& name, std::vector<uint32_t> dims, uint32_t precision) {
  auto tensor = std::make_unique<Tensor>(0, name, dims, precision, true);
  const uint32_t id = tensor->get_id();
  model.add_tensor(std::move(tensor));
  return id;
}

/* DRAM addresses written by MOVOUT */
std::vector<addr_type> stored_lines(Operation& op) {
  std::vector<addr_type> lines;
  for (const auto& tile : op.get_tiles())
    for (const auto& inst : tile->instructions)
      if (inst->opcode == Opcode::MOVOUT)
        lines.insert(lines.end(), inst->src_addrs.begin(), inst->src_addrs.end());
  return lines;
}

void expect_inside(const std::vector<addr_type>& lines, Tensor* tensor) {
  for (addr_type line : lines) {
    EXPECT_GE(line, tensor->get_address());
    EXPECT_LT(line, tensor->get_address() + tensor->get_size());
  }
}
}  // namespace

/* Llama MLP: input is the fused gate/up output (tokens x 2I), output is tokens x I */
TEST(LanguageOpTest, SwigluOutputAddress) {
  SimulationConfig config = language_op_config();
  Model model(json::object(), config, "llama-mlp");
  const uint32_t tokens = 16, intermediate = 64;
  const uint32_t input = add_tensor(model, "gate_up", {tokens, 2 * intermediate}, config.precision);
  const uint32_t bias = add_tensor(model, "gate_up_bias", {2 * intermediate}, config.precision);
  std::map<std::string, std::string> attributes = {
      {"has_bias", "1"}, {"activation", "swish"}, {"llama_mlp", "1"}};
  BiasAct op(config, &model, "act", attributes);
  op.add_input(input);
  op.add_input(bias);
  MappingTable mapping_table(config);
  op.initialize_tiles(mapping_table);

  Tensor* output = op.get_output(0);
  ASSERT_EQ(output->get_dims(), (std::vector<uint32_t>{tokens, intermediate}));
  const auto lines = stored_lines(op);
  EXPECT_EQ(lines.size() * config.dram_req_size, uint64_t(tokens) * intermediate * config.precision);
  expect_inside(lines, output);
  delete[] config.core_config;
}

/* The result goes to the output tensor, not back to the input */
TEST(LanguageOpTest, SkipLayerNormOutputAddress) {
  SimulationConfig config = language_op_config();
  Model model(json::object(), config, "layernorm");
  const uint32_t tokens = 16, hidden = 64;
  const uint32_t input = add_tensor(model, "residual", {tokens, hidden}, config.precision);
  const uint32_t skip = add_tensor(model, "projection", {tokens, hidden}, config.precision);
  const uint32_t weight = add_tensor(model, "ln_weight", {hidden}, config.precision);
  const uint32_t bias = add_tensor(model, "ln_bias", {hidden}, config.precision);
  std::map<std::string, std::string> attributes;
  SkipLayerNorm op(config, &model, "ln", attributes);
  for (uint32_t id : {input, skip, weight, bias}) op.add_input(id);
  MappingTable mapping_table(config);
  op.initialize_tiles(mapping_table);

  const auto lines = stored_lines(op);
  EXPECT_EQ(lines.size() * config.dram_req_size, uint64_t(tokens) * hidden * config.precision);
  expect_inside(lines, op.get_output(0));
  delete[] config.core_config;
}
