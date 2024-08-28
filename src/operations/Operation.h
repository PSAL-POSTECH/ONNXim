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
            uint32_t id, uint32_t target_core);
  Operation(SimulationConfig config, MappingTable& mapping_table, uint32_t target_core);
  Operation(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core);
  Operation(const Operation& operation);
  Operation(SimulationConfig config, Model* model,
            std::string name,  std::map<std::string, std::string>&attribute, uint32_t target_core);
  virtual ~Operation() = default;
  virtual void set_finish();

  virtual std::string get_name() { return _name; }
  virtual std::string get_optype() { return _optype; }
  virtual uint32_t get_id() { return _id; }
  virtual uint32_t num_inputs() { return _inputs.size(); }
  virtual Tensor* get_input(int id);
  virtual void add_input(int id);
  virtual void add_output(int id);
  virtual uint32_t num_outputs() { return _outputs.size(); }
  virtual Tensor* get_output(int id);
  virtual void set_model(Model* model) { _model=model; }
  virtual std::vector<uint32_t> get_child_nodes();
  virtual std::deque<std::unique_ptr<Tile>>& get_tiles();
  virtual void clear_tiles();
  virtual void initialize_tiles(MappingTable& mapping_table) = 0;
  virtual bool check_executable();
  bool check_finish() { return _finish; };
  uint32_t target_core=0; // Targeted core id

 protected:
  virtual void initialize_instructions(Tile* tile, Mapping mapping) {}
  addr_type make_address(std::vector<uint32_t> index, std::vector<uint32_t> dims);
  addr_type get_operand_addr(uint32_t operand_id);
  std::string get_attribute(std::string key);
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
  std::deque<std::unique_ptr<Tile>> _tiles;
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