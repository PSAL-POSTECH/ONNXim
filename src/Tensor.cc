#include "Tensor.h"

#include "Model.h"
#include "operations/Operation.h"

#include <unordered_map>
#include <mutex>

// Define globals
std::unordered_map<uint32_t, TensorInfo> g_tensor_addr_map;
std::mutex g_tensor_map_mutex;

Tensor::Tensor(uint32_t src_node, onnx::TensorProto &tensor_proto, int precision,
               bool produced = false) {
  _id = generate_id();
  _temporal = false;
  _src_node = src_node;
  _name = tensor_proto.name();
  for (int dim : tensor_proto.dims()) {
    _dims.push_back(dim);
  }
  spdlog::trace("Tensor: {}", _name);
  _produced = produced;
  _precision = precision;
  allocate_tensor(precision);
}

Tensor::Tensor(uint32_t src_node, std::string name, std::vector<uint32_t> &dims,
               int precision, bool produced = false) {
  _temporal = false;
  _id = generate_id();
  _src_node = src_node;
  _name = name;
  for (int dim : dims) {
    _dims.push_back(dim);
  }
  spdlog::trace("Tensor: {} {}", _name, dims);
  _produced = produced;
  _precision = precision;
  allocate_tensor(precision);
}

Tensor::Tensor(const Tensor &tensor) {
  _temporal = false;
  _produced = tensor._produced;
  _id = tensor._id;
  _name = tensor._name;
  _dims = tensor._dims;
  _src_node = tensor._src_node;
  _child_nodes = tensor._child_nodes;
  _address = tensor._address;
  _size = tensor._size;
  _precision = tensor._precision;
}

Tensor::Tensor(uint32_t src_node, std::string name, int precision) {
  //Temproal definition, need to define
  _temporal = true;
  _id = generate_id();
  _src_node = src_node;
  _name = name;
  _precision = precision;
}

void Tensor::define_tensor(addr_type address, std::vector<uint32_t> &dims) {
  if (_dims.empty()) {
    _temporal = false;
    _address = address;
    _size = _precision;
    for (int dim : dims) {
      _dims.push_back(dim);
      _size *= dim;
    }
  } else {
    throw("Error: cannot redefine already created tensor");
  }
}

void Tensor::redefine_tensor(uint32_t id, std::vector<uint32_t> &dims) {
  if (_dims.empty()) {
    _src_node = id;
    _size = _precision;
    for (int dim : dims) {
      _dims.push_back(dim);
      _size *= dim;
    }
  } else {
    bool condition = false;
    if (_dims.size() == dims.size() && id == _src_node) {
      condition = true;
      for (int i = 0; i < _dims.size(); i++) {
        condition = condition && (_dims[i] == dims[i]);
      }
    }
    if (!condition) throw("Error: cannot redefine already created tensor");
  }
}

void Tensor::resize_tensor(std::vector<uint32_t> &dims) {
  _dims.clear();
  _size = _precision;
  for (int dim : dims) {
    _dims.push_back(dim);
    _size *= dim;
  }
}

void Tensor::add_child_node(Operation *op) {
  _child_nodes.push_back(op->get_id());
}

void Tensor::allocate_tensor(int precision)
{
    uint32_t size = 1;
    for (auto dim : _dims)
        size *= dim;

    _address = allocate_address(size * precision);
    _size = size * precision;

    // Register tensor info globally
    std::lock_guard<std::mutex> lock(g_tensor_map_mutex);
    g_tensor_addr_map[_id] = {
        _name,
        static_cast<uint64_t>(_address),
        static_cast<uint64_t>(_address + _size),
        static_cast<uint64_t>(_size)
    };

    spdlog::debug("Tensor registered: id={} name={} addr=0x{:x}-0x{:x} size={}",
                  _id, _name, _address, _address + _size, _size);
}

void Tensor::print_tensor() {
  spdlog::info("Tensor: {} {} {} {}", _name, _src_node, _dims, _size);
}

void log_tensor_allocation_table() {
    std::lock_guard<std::mutex> lock(g_tensor_map_mutex);

    spdlog::info("{:<10} {:<20} {:<18} {:<18} {:<12}", 
                 "TensorID", "Name", "StartAddr", "EndAddr", "Size");

    for (auto &entry : g_tensor_addr_map) {
        const auto &info = entry.second;
        spdlog::info("{:<10} {:<20} 0x{:016x} - 0x{:016x} {:<12}", 
                     entry.first, info.name, info.start_addr, info.end_addr, info.size);
    }
}