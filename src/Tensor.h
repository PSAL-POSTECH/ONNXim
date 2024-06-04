#pragma once
#include "Common.h"

class Model;
class Operation;

class Tensor {
 public:
  Tensor(uint32_t src_node, onnx::TensorProto &tensor_proto, int precision, bool produced);
  Tensor(uint32_t src_node, std::string name, std::vector<uint32_t> &dims,
         int precision, bool produced);
  Tensor(uint32_t src_node, std::string name, int precision);
  Tensor(const Tensor &tensor);

  void define_tensor(addr_type address, std::vector<uint32_t> &dims);
  void redefine_tensor(uint32_t src_node, std::vector<uint32_t> &dims);
  void resize_tensor(std::vector<uint32_t> &dims);
  void add_child_node(Operation *op);

  uint32_t get_id() { return _id; }
  std::string get_name() { return _name; }
  uint32_t get_src_node() { return _src_node; }
  std::vector<uint32_t> get_dims() { return _dims; }
  void set_produced() { _produced = true; }
  bool get_produced() { return _produced; }
  uint32_t num_child_nodes() { return _child_nodes.size(); }
  uint32_t get_child_node(uint32_t id) { return _child_nodes[id]; }

  void allocate_tensor(int precision);
  addr_type get_address() { return _address; }
  uint64_t get_size() { return _size; }
  void print_tensor();

 private:
  bool _temporal;
  uint32_t _precision;
  bool _produced;
  uint32_t _id;
  std::string _name;
  std::vector<uint32_t> _dims;
  uint32_t _src_node;
  std::vector<uint32_t> _child_nodes;
  addr_type _address;
  uint64_t _size;
  friend Model;
};