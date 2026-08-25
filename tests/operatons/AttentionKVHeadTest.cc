#include <algorithm>
#include <memory>
#include <set>

#include "Common.h"
#include "Model.h"
#include "SimulationConfig.h"
#include "gtest/gtest.h"
#include "operations/Attention.h"
#define ATTENTION_SEQ_LEN 64
#define ATTENTION_NUM_TOKENS 8

SimulationConfig get_default_attention_config() {
  SimulationConfig config;
  config.num_cores = 1;
  config.core_freq = 1000;
  config.core_config = new struct CoreConfig;
  config.core_config[0].core_type = CoreType::SYSTOLIC_WS;
  config.core_config[0].core_height = 8;
  config.core_config[0].core_width = 8;
  config.core_config[0].vector_process_bit = 1024;
  config.core_config[0].spad_size = 2048;
  config.core_config[0].accum_spad_size = 1024;
  config.precision = 1;
  config.dram_req_size = 32;
  config.dram_freq = 1000;
  config.dram_channels = 16;
  config.layout = "NHWC";
  config.core_print_interval = 1000000;
  return config;
}

std::map<std::string, std::string> make_attention_attributes(
    uint32_t num_heads, uint32_t num_kv_heads, uint32_t dk) {
  return {{"num_tokens", std::to_string(ATTENTION_NUM_TOKENS)},
          {"num_heads", std::to_string(num_heads)},
          {"num_kv_heads", std::to_string(num_kv_heads)},
          {"hidden_size", std::to_string(num_heads * dk)}};
}

/* Registers the output tensor and the shapes that initialize_tiles() would
   otherwise derive from the input tensors of an already built graph. */
void define_attention_tensors(Attention& op, Model& model,
                              SimulationConfig config, uint32_t num_heads,
                              uint32_t num_kv_heads, uint32_t dk) {
  std::vector<uint32_t> output_dims = {ATTENTION_NUM_TOKENS, num_heads * dk};
  std::unique_ptr<Tensor> output = std::make_unique<Tensor>(
      0, "attention_output", output_dims, config.precision, true);
  op.add_output(output->get_id());
  model.add_tensor(std::move(output));

  op._seq = ATTENTION_SEQ_LEN;
  op._query_shape = {num_heads, ATTENTION_NUM_TOKENS, dk};
  op._key_shape = {num_kv_heads, ATTENTION_SEQ_LEN, dk};
  op._value_shape = op._key_shape;
}

/* A tile loads its key cache first. Those source addresses come from
   make_address({kv_head, seq, d}, key_shape), so the lowest of them tells which
   KV head the tile reads. */
uint32_t kv_head_read_by(Tile& tile, addr_type kv_head_stride) {
  Instruction& key_load = *tile.instructions.front();
  EXPECT_EQ(key_load.opcode, Opcode::MOVIN);
  return *std::min_element(key_load.src_addrs.begin(),
                           key_load.src_addrs.end()) /
         kv_head_stride;
}

std::set<uint32_t> kv_heads_read(Attention& op, SimulationConfig config) {
  addr_type kv_head_stride = ATTENTION_SEQ_LEN * op._dk * config.precision;
  std::set<uint32_t> kv_heads;
  for (std::unique_ptr<Tile>& tile : op.get_tiles())
    kv_heads.insert(kv_head_read_by(*tile, kv_head_stride));
  return kv_heads;
}

std::set<uint32_t> every_kv_head(uint32_t num_kv_heads) {
  std::set<uint32_t> kv_heads;
  for (uint32_t kv_head = 0; kv_head < num_kv_heads; kv_head++)
    kv_heads.insert(kv_head);
  return kv_heads;
}

/* Multi head attention: every query head owns one KV head, so the tiles have to
   cover all of them. Deriving kv_head_idx from the KV head count instead of the
   query heads per KV head collapsed every tile onto KV head 0. */
TEST(AttentionMhaKvHeadMapping, BasicAssertions) {
  uint32_t num_heads = 12, num_kv_heads = 12, dk = 64;
  SimulationConfig config = get_default_attention_config();
  Model model(json::object(), config, "attention_mha");
  std::map<std::string, std::string> attributes =
      make_attention_attributes(num_heads, num_kv_heads, dk);

  Attention op(config, &model, "attention", attributes, 0);
  define_attention_tensors(op, model, config, num_heads, num_kv_heads, dk);
  MappingTable mapping_table = MappingTable(config);
  op.initialize_tiles(mapping_table);

  EXPECT_EQ(kv_heads_read(op, config), every_kv_head(num_kv_heads));
}

/* Grouped query attention: four query heads share one KV head. The same wrong
   divisor mapped the eight groups onto KV heads 0..3 and never read 4..7. */
TEST(AttentionGqaKvHeadMapping, BasicAssertions) {
  uint32_t num_heads = 32, num_kv_heads = 8, dk = 128;
  SimulationConfig config = get_default_attention_config();
  Model model(json::object(), config, "attention_gqa");
  std::map<std::string, std::string> attributes =
      make_attention_attributes(num_heads, num_kv_heads, dk);

  Attention op(config, &model, "attention", attributes, 0);
  define_attention_tensors(op, model, config, num_heads, num_kv_heads, dk);
  MappingTable mapping_table = MappingTable(config);
  op.initialize_tiles(mapping_table);

  EXPECT_EQ(kv_heads_read(op, config), every_kv_head(num_kv_heads));
}

/* The mapping is computed a second time in the initialize_instructions()
   overload used by initialize_onnx_tiles(), which passes one query head per
   group. Driven directly because building an ONNX graph needs a model file. */
TEST(AttentionOnnxPathKvHeadMapping, BasicAssertions) {
  uint32_t num_heads = 12, num_kv_heads = 12, dk = 64;
  SimulationConfig config = get_default_attention_config();
  Model model(json::object(), config, "attention_onnx_path");
  std::map<std::string, std::string> attributes =
      make_attention_attributes(num_heads, num_kv_heads, dk);

  Attention op(config, &model, "attention", attributes, 0);
  define_attention_tensors(op, model, config, num_heads, num_kv_heads, dk);
  addr_type kv_head_stride = ATTENTION_SEQ_LEN * dk * config.precision;

  std::set<uint32_t> kv_heads;
  for (uint32_t head_idx = 0; head_idx < num_heads; head_idx++) {
    Tile tile{.status = Tile::Status::INITIALIZED};
    op.initialize_instructions(&tile, head_idx, 1);
    kv_heads.insert(kv_head_read_by(tile, kv_head_stride));
  }

  EXPECT_EQ(kv_heads, every_kv_head(num_kv_heads));
}
