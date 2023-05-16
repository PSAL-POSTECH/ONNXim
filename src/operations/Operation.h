#pragma once

#include "../Common.h"
#include "../Mapping.h"
#include "../Tensor.h"

class Model;
class OpParser;

// Graph Node
class Operation {
 public:
  Operation(SimulationConfig config, Model* model, onnx::NodeProto& node_proto,
            uint32_t id);
  Operation(SimulationConfig config, MappingTable mapping_table);
  Operation(SimulationConfig config, Model* model, onnx::NodeProto& node_proto);
  Operation(const Operation& operation);
  virtual void set_finish();

  virtual std::string get_name() { return _name; }
  virtual std::string get_optype() { return _optype; }
  virtual uint32_t get_id() { return _id; }
  virtual uint32_t num_inputs() { return _inputs.size(); }
  virtual Tensor* get_input(int id);
  virtual uint32_t num_outputs() { return _outputs.size(); }
  virtual Tensor* get_output(int id);
  virtual std::vector<uint32_t> get_child_nodes();
  virtual std::deque<Tile> get_tiles();
  virtual void initialize_tiles(MappingTable mapping_table) = 0;
  virtual bool check_executable();
  bool check_finish() { return _finish; };

 protected:
  virtual void initialize_instructions(Tile& tile, Mapping mapping) {}
  addr_type get_operand_addr(uint32_t operand_id);
  addr_type make_activation_address(uint32_t N, uint32_t H, uint32_t W,
                                    uint32_t C, std::vector<uint32_t> shape);
  addr_type make_weight_address(uint32_t S, uint32_t R, uint32_t M, uint32_t C,
                                std::vector<uint32_t> shape);

 protected:
  static const uint32_t _NO_OPERAND = 0;
  static const uint32_t _INPUT_OPERAND = 100;
  static const uint32_t _OUTPUT_OPERAND = 200;
  uint32_t _id;
  std::string _name;
  std::string _optype;
  SimulationConfig _config;
  Model* _model;
  onnx::NodeProto _proto;
  std::vector<uint32_t> _inputs;
  std::vector<uint32_t> _outputs;
  std::map<std::string, std::string> _attributes;
  std::deque<Tile> _tiles;
  std::vector<std::vector<std::vector<addr_type>>> _weight_addrs;
  std::vector<std::vector<std::vector<std::vector<addr_type>>>> _input_addrs;
  std::vector<std::vector<std::vector<std::vector<addr_type>>>> _output_addrs;

  int Ndim;    // Batch dimension of activation tensor (commoly 0)
  int Hdim;    // Height dimension of activation tensor
  int Wdim;    // Width dimension of activation tensor
  int Cdim;    // Channel dimension of activation tensor
  int Cdim_w;  // Channel dimension of weight tensor
  int Mdim;    // Output channel dimension of weight tensor
  int Sdim;    // Height dimension of weight tensor
  int Rdim;    // Width dimension of weight tensor

  bool _finish;
  friend Model;
};