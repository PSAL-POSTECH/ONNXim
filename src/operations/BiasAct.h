#pragma once
#include "Operation.h"

class BiasAct : public Operation {
 public:
  BiasAct(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
  BiasAct(SimulationConfig config, Model* model, std::string name,
          std::map<std::string, std::string>& attributes, uint32_t target_core=0);

  void initialize_tiles(MappingTable& mapping_table) override;

 private:
  void calculate_loops();
  void initialize_instructions(Tile* tile, Mapping mapping,
                               uint32_t token_offset, uint32_t tokens);
  std::vector<uint32_t> _bias_shape;

  std::vector<uint32_t> _input_shape;
  std::vector<uint32_t> _output_shape;

  uint32_t _batch_size;
  uint32_t _seq;
  uint32_t _dk;
  uint32_t _tokens_per_tile;
  bool _llama_mlp;
  bool _use_bias;
  Opcode _activation;
};