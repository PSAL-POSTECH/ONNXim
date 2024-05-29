#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "Model.h"
#include "operations/OperationFactory.h"
#include "operations/Attention.h"

Model::Model(std::string onnx_path, json model_config, SimulationConfig config, std::string name, MappingTable& mapping_table) {
  _onnx_path = onnx_path;
  _id = generate_id();
  _name = name;
  _root_node_id = generate_id();
  _config = config;
  _model_config = model_config;
  if (_model_config.contains("request_time"))
    _request_time = uint64_t(double(_model_config["request_time"]) * 1000 * 1000 * 1000); // Pico seconds
  else
    _request_time = 0;
  _mapping_table = mapping_table;
  if (_model_config.contains("partition_id")) {
    _partition_id = uint32_t(_model_config["partition_id"]);
  }
}

Model::Model(json model_config, SimulationConfig config, std::string name)
      :_model_config(model_config), _config(config), _name(name) {
  _id = generate_id(); 
  _mapping_table = MappingTable(_config);
}

Tensor* Model::get_tensor(uint32_t id) {
  return _tensor_map[id].get();
}

Tensor* Model::find_tensor(std::string name) {
  for(auto const& [key, val]: _tensor_map) {
    if(val->_name == name) {
      return val.get();
    }
  }
  return nullptr;
}

void Model::add_tensor(std::unique_ptr<Tensor> edge) {
  _tensor_map[edge->get_id()] = std::move(edge);
}

void Model::initialize_model(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  auto start = std::chrono::high_resolution_clock::now();

  onnx::ModelProto model_proto;
  std::vector<std::unique_ptr<Tensor>> input_tensors;
  std::ifstream model_istream(_onnx_path);
  if (!model_istream) {
    spdlog::error("Error opening file: {}", _onnx_path);
    exit(EXIT_FAILURE);
  }
  google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
  model_proto.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
  auto input = model_proto.graph().input();

  for (auto iter: input) {
    std::vector<uint32_t> input_dim;
    std::string input_name = iter.name();
    auto input_shape = iter.type().tensor_type().shape();

    /* Parsing input tensor shape */
    for (int dim_idx=0; dim_idx<input_shape.dim_size(); dim_idx++) {
      /* Get axis, dynamic axis */
      int dim_value = input_shape.dim(dim_idx).dim_value();
      std::string dim_param = input_shape.dim(dim_idx).dim_param();
      spdlog::debug("input name: {} val: {} param: {}", input_name, dim_value, dim_param);
      if (dim_value==0 && dim_param!="") {
        /* Check local axis map */
        if (_axis_map.find(dim_param) != _axis_map.end())
          input_dim.push_back(_axis_map[dim_param]);
        else {
          /* Fallback to config file and read dynamic axis */
          input_dim.push_back(_model_config[dim_param]);
          _axis_map[dim_param] = _model_config[dim_param];
        }
      } else {
        input_dim.push_back(dim_value);
      }
    }

    /* NCHW to NHWC convert */
    if (input.size()==1 && input_dim.size()==4 && input_dim.at(2)==input_dim.at(3)) {
      uint32_t channel = input_dim.at(1);
      input_dim.erase(input_dim.begin() + 1);
      input_dim.push_back(channel);
    }

    auto input_tensor = std::make_unique<Tensor>(_root_node_id, input_name, input_dim, _config.precision * 16, true);
    int id = input_tensor->get_id();
    input_tensor->set_produced();
    _tensor_map[id] = std::move(input_tensor);
  }

  for(auto it = weight_table.begin(); it != weight_table.end(); it++) {
    //initialize weights
    auto tensor = std::make_unique<Tensor>(*it->get());
    tensor->set_produced();
    uint32_t id = tensor->get_id();
    _tensor_map[id] = std::move(tensor);
  }


  for(auto node_proto : model_proto.graph().node()) {
    auto node = OperationFactory::create_operation(this, node_proto);
    if(node != nullptr) {
      int node_id = node->get_id();
      _operation_map[node->get_id()] = std::move(node);
      /* For skipping attention block */
      if (node_proto.op_type() == "SkipLayerNormalization" && _model_config.contains("nr_atten"))
        if (_model_config["nr_atten"] != -1 && ++nr_skip >= int(_model_config["nr_atten"])*2) {
          _operation_map[node_id].get()->_outputs.clear();
          break;
        }
    }
  }

  for(auto& [key, val] : _operation_map) {
    /* Attention is speacial case */
    if (val->get_optype() == "Attention") {
      Attention* attention_node = static_cast<Attention*>(val.get());
      attention_node->initialize_onnx_tiles(_mapping_table);
      int projection_id = attention_node->_projection_node->get_id();
      _operation_map[projection_id] = std::move(std::unique_ptr<GemmWS>(attention_node->_projection_node));
      _operation_map[projection_id]->initialize_tiles(_mapping_table);
    }
    else {
      val->initialize_tiles(_mapping_table);
    }
  }

  for (auto& [key, val]: _operation_map) {
    if(val->check_executable()) {
      spdlog::debug("runnable op, {}", val->get_optype());
      _executable_layer.push_back(val.get());
    } 
  }

  /* Model initialization time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("{} Model initialization time: {:2f} seconds", _onnx_path, duration.count());
}

void Model::initialize_weight(std::vector<std::unique_ptr<Tensor>>& weight_table) {
  spdlog::info("Model Name");

  weight_table.clear();
  onnx::ModelProto model_proto;
  std::ifstream model_istream(_onnx_path);
  if (!model_istream) {
    spdlog::error("Error opening file: {}", _onnx_path);
    exit(EXIT_FAILURE);
  }
  google::protobuf::io::IstreamInputStream zero_copy_input(&model_istream);
  model_proto.ParseFromZeroCopyStream(&zero_copy_input) && model_istream.eof();
 for(auto initializer : model_proto.graph().initializer()) {
    //initialize weights
    auto tensor = std::make_unique<Tensor>(_root_node_id, initializer, _config.precision, true);
    tensor->set_produced();
    weight_table.push_back(std::move(tensor));

  }
}


void Model::set_layer_finish(uint32_t id) {
  _operation_map[id]->set_finish();
  for(auto op_id : _operation_map[id]->get_child_nodes()) {
    Operation* op = _operation_map[op_id].get();
    if(op->check_executable() && !check_exist_in_exeutable(op->get_id()))  {
      _executable_layer.push_back(op);
    }
  }

}

uint32_t Model::executable_layer_size() {
  return _executable_layer.size();
}

Operation* Model::get_executable_tile() {
  Operation* op = nullptr;
  if (_executable_layer.size()){
    op = _executable_layer.front();
    _executable_layer.erase(_executable_layer.begin());
  }
  return op;
}

void Model::update_start_time(uint64_t start_time) {
  if (!_started) {
    _start_time = start_time;
    _started = true;
  }
}

bool Model::check_finish() {
  bool finish = true;
  for(auto const& [key, val] : _operation_map) {
    finish = finish && val->check_finish();
  }
  return finish;
}

bool Model::check_exist_in_exeutable(uint32_t op_id) {
  for(auto iter = _executable_layer.begin(); iter != _executable_layer.end(); iter ++) {
    if(op_id == (*iter)->get_id()) {
      return true;
    }
  }
  return false;
}

bool Model::check_regressive() {
  if (_axis_map.find("total_seq_len") == _axis_map.end()){
    spdlog::info("No total_seq_len!");
    return false;
  }
  if ((_axis_map["total_seq_len"]+1) == _model_config["output_seq_len"]) {
    spdlog::info("Reached output seq len");
    return false;
  }
  return true;
}

void Model::prepare_regressive() {
  /* This method should be called when check_regressive() is true */
  if (_axis_map["past_seq_len"] == 0) {
    _axis_map["past_seq_len"] = _axis_map["seq_len"];
    _axis_map["seq_len"] = 1;
    _axis_map["total_seq_len"]++;
  } else {
    _axis_map["past_seq_len"]++;
    _axis_map["seq_len"] = 1;
    _axis_map["total_seq_len"]++;
  }

  _operation_map.clear();
  _tensor_map.clear();
  _executable_layer.clear();
  nr_skip = 0;
  _start_time = 0;
  _started = false;
}