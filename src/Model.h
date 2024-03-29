#ifndef INSTRUCTION_H
#define INSTRUCTION_H

#include "Common.h"
#include "helper/HelperFunctions.h"
#include "operations/Operation.h"
#include "Tensor.h"
#include "Mapping.h"
class Model {
  public:
    Model(std::string onnx_path, json model_config, SimulationConfig config, std::string name);
    Model(const Model& model);

    Tensor* get_tensor(uint32_t id);
    Tensor* find_tensor(std::string name);
    void add_tensor(std::unique_ptr<Tensor> tensor);
    void initialize_model(MappingTable& mapping_table);
    void set_layer_finish(uint32_t id); 

    std::string get_name() { return _name; }
    std::vector<Tensor*> get_input_tensor() { return _input_tensor; }
    std::vector<Operation*> get_executable_layers();
    bool check_finish();
  private:
    json _model_config;
    std::string _name;
    onnx::ModelProto _model_proto;
    uint32_t _root_node_id;
    std::vector<Tensor*> _input_tensor;
    std::map<uint32_t, std::unique_ptr<Operation>> _operation_map;
    std::map<uint32_t, std::unique_ptr<Tensor>> _tensor_map;
    std::vector<Operation*> _executable_layer;
    SimulationConfig _config;

    /* Number of simulating attention block */
    int nr_skip = 0; // NR_SKIP == 2 * NR_ATTEN

    bool check_exist_in_exeutable(uint32_t id);
};

#endif