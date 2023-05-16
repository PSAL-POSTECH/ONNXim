#ifndef INSTRUCTION_H
#define INSTRUCTION_H

#include "Common.h"
#include "helper/HelperFunctions.h"
#include "operations/Operation.h"
#include "Tensor.h"
#include "Mapping.h"
class Model {
  public:
    Model(std::string onnx_path, SimulationConfig config, std::string name);
    Model(const Model& model);

    Tensor* get_tensor(uint32_t id);
    Tensor* find_tensor(std::string name);
    void add_tensor(std::unique_ptr<Tensor> tensor);
    void initialize_model(std::string input_names, std::vector<uint32_t> &input_dims, MappingTable mapping_table);
    // void initialize_model(std::vector<std::string> &input_names, std::vector<std::vector<uint32_t>> &input_dims);
    void set_layer_finish(uint32_t id); 

    std::string get_name() { return _name; }
    Tensor* get_input_tensor() { return _input_tensor; }
    std::vector<Operation*> get_executable_layers();
    bool check_finish();
  private:
    std::string _name;
    std::string _input_name;
    std::vector<uint32_t> _input_dim;
    onnx::ModelProto _model_proto;
    uint32_t _root_node_id;
    Tensor* _input_tensor;
    std::map<uint32_t, std::unique_ptr<Operation>> _operation_map;
    std::map<uint32_t, std::unique_ptr<Tensor>> _tensor_map;
    std::vector<Operation*> _executable_layer;
    SimulationConfig _config;

    bool check_exist_in_exeutable(uint32_t id);
};

#endif