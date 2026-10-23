#include <memory>
#include <set>

#include "Common.h"
#include "Model.h"
#include "SimulationConfig.h"
#include "gtest/gtest.h"
#include "operations/Attention.h"

namespace {

constexpr uint32_t kSequenceLength = 64;
constexpr uint32_t kNumTokens = 8;

struct HeadLayout {
  uint32_t query_heads;
  uint32_t kv_heads;
  uint32_t head_dimension;
  uint32_t precision;
};

void PrintTo(const HeadLayout& layout, std::ostream* stream) {
  *stream << layout.query_heads << " query heads, " << layout.kv_heads
          << " KV heads, dimension " << layout.head_dimension
          << ", " << layout.precision << " bytes per element";
}

class TestAttention : public Attention {
 public:
  using Attention::Attention;
  using Attention::_INPUT_OPERAND;
};

class AttentionKvHeadMappingTest : public ::testing::TestWithParam<HeadLayout> {
 protected:
  CoreConfig core_{};
  SimulationConfig config_{};
  std::unique_ptr<Model> model_;
  std::unique_ptr<TestAttention> attention_;

  void SetUp() override {
    core_.core_type = CoreType::SYSTOLIC_WS;
    core_.core_height = 8;
    core_.core_width = 8;
    core_.vector_process_bit = 1024;
    core_.sram_width = 32;
    core_.spad_size = 2048;
    core_.accum_spad_size = 1024;
    config_.num_cores = 1;
    config_.core_freq = 1000;
    config_.core_config = &core_;
    config_.precision = GetParam().precision;
    config_.dram_req_size = 32;
    config_.dram_freq = 1000;
    config_.dram_channels = 16;
    config_.layout = "NHWC";
    config_.core_print_interval = 1000000;

    model_ = std::make_unique<Model>(json::object(), config_, "attention_test");
    std::map<std::string, std::string> attributes = {
        {"num_tokens", std::to_string(kNumTokens)},
        {"num_heads", std::to_string(GetParam().query_heads)},
        {"num_kv_heads", std::to_string(GetParam().kv_heads)},
        {"hidden_size", std::to_string(GetParam().query_heads * GetParam().head_dimension)}};
    attention_ = std::make_unique<TestAttention>(
        config_, model_.get(), "attention", attributes, 0);

    AddInput("query", {kNumTokens, GetParam().query_heads * GetParam().head_dimension});
    AddInput("key", {kSequenceLength, GetParam().kv_heads * GetParam().head_dimension});
    AddInput("value", {kSequenceLength, GetParam().kv_heads * GetParam().head_dimension});
    ASSERT_NE(attention_->get_input(1)->get_address(), 0u);
    ASSERT_NE(attention_->get_input(2)->get_address(), 0u);
    ASSERT_NE(attention_->get_input(1)->get_address(),
              attention_->get_input(2)->get_address());
  }

  void AddInput(const std::string& name, std::vector<uint32_t> dims) {
    auto tensor = std::make_unique<Tensor>(0, name, dims, config_.precision, true);
    const uint32_t id = tensor->get_id();
    model_->add_tensor(std::move(tensor));
    attention_->add_input(id);
  }

  void ExpectCacheLoads(const Tile& tile, uint32_t expected_kv_head) {
    const addr_type head_bytes = kSequenceLength * GetParam().head_dimension * config_.precision;
    for (uint32_t input = 1; input <= 2; ++input) {
      SCOPED_TRACE(input == 1 ? "key" : "value");
      const addr_type base = attention_->get_input(input)->get_address();
      std::set<addr_type> expected;
      for (addr_type offset = 0; offset < head_bytes; offset += config_.dram_req_size)
        expected.insert(base + expected_kv_head * head_bytes + offset);

      uint32_t loads = 0;
      for (const auto& instruction : tile.instructions) {
        if (instruction->opcode != Opcode::MOVIN ||
            instruction->operand_id != TestAttention::_INPUT_OPERAND + input)
          continue;
        ++loads;
        ASSERT_FALSE(instruction->src_addrs.empty());
        EXPECT_EQ(instruction->size, expected.size());
        EXPECT_EQ(instruction->src_addrs.size(), expected.size());
        EXPECT_EQ(std::set<addr_type>(instruction->src_addrs.begin(),
                                     instruction->src_addrs.end()), expected);
      }
      EXPECT_EQ(loads, 1u);
    }
  }
};

TEST_P(AttentionKvHeadMappingTest, GeneratedTilesReadMatchingKeyAndValueHeads) {
  MappingTable mapping_table(config_);
  attention_->initialize_tiles(mapping_table);
  const auto& tiles = attention_->get_tiles();
  // Keep the sequence and query within a single tile to isolate head mapping.
  ASSERT_EQ(tiles.size(), GetParam().kv_heads);
  for (uint32_t kv_head = 0; kv_head < GetParam().kv_heads; ++kv_head) {
    SCOPED_TRACE(kv_head);
    ASSERT_EQ(tiles[kv_head]->batch, kv_head);
    ExpectCacheLoads(*tiles[kv_head], kv_head);
  }
}

TEST_P(AttentionKvHeadMappingTest, DirectInstructionsReadMatchingKeyAndValueHeads) {
  // Exercise the overload used by the ONNX path without requiring a model file.
  // Grouped cases test this helper, not end-to-end ONNX GQA support.
  attention_->_seq = kSequenceLength;
  attention_->_query_shape = {GetParam().query_heads, kNumTokens, GetParam().head_dimension};
  attention_->_key_shape = {GetParam().kv_heads, kSequenceLength, GetParam().head_dimension};
  attention_->_value_shape = attention_->_key_shape;
  std::vector<uint32_t> dims = {kNumTokens, GetParam().query_heads * GetParam().head_dimension};
  auto output = std::make_unique<Tensor>(0, "output", dims, config_.precision, true);
  attention_->add_output(output->get_id());
  model_->add_tensor(std::move(output));

  attention_->calculate_loops();
  ASSERT_EQ(attention_->_heads_per_tile.size(), 1u);
  const uint32_t heads_per_group = attention_->_heads_per_tile.front();
  ASSERT_EQ(heads_per_group, GetParam().query_heads / GetParam().kv_heads);
  for (uint32_t kv_head = 0; kv_head < GetParam().kv_heads; ++kv_head) {
    SCOPED_TRACE(kv_head);
    Tile tile{};
    tile.status = Tile::Status::INITIALIZED;
    attention_->initialize_instructions(&tile, kv_head * heads_per_group,
                                        heads_per_group);
    ExpectCacheLoads(tile, kv_head);
  }
}

INSTANTIATE_TEST_CASE_P(
    HeadLayouts, AttentionKvHeadMappingTest,
    ::testing::Values(
        HeadLayout{12, 12, 64, 1}, HeadLayout{32, 8, 128, 1},
        HeadLayout{28, 4, 128, 1}, HeadLayout{32, 1, 64, 1},
        HeadLayout{16, 4, 64, 1}, HeadLayout{1, 1, 64, 1},
        HeadLayout{12, 12, 64, 2}, HeadLayout{32, 8, 128, 2},
        HeadLayout{28, 4, 128, 2}, HeadLayout{32, 1, 64, 2},
        HeadLayout{16, 4, 64, 2}, HeadLayout{1, 1, 64, 2}));

}  // namespace
