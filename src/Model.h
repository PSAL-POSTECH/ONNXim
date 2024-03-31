#ifndef INSTRUCTION_H
#define INSTRUCTION_H

#include "Common.h"
#include "helper/HelperFunctions.h"
#include "operations/Operation.h"
#include "Tensor.h"
#include "Mapping.h"
class Model {
  public:
    Model(std::string onnx_path, json model_config, SimulationConfig config, std::string name, MappingTable& map);
    Model(const Model& model);

    Tensor* get_tensor(uint32_t id);
    Tensor* find_tensor(std::string name);
    void add_tensor(std::unique_ptr<Tensor> tensor);
    void initialize_model();
    void set_layer_finish(uint32_t id); 

    std::string get_name() { return _name; }
    uint32_t executable_layer_size();
    Operation* get_executable_tile();
    uint64_t get_request_time() const { return _request_time; }
    void set_request_time(uint64_t request_time) { _request_time=request_time; }
    uint64_t get_start_time() const { return _start_time; }
    void update_start_time(uint64_t start_time);
    bool check_finish();
    uint32_t get_partition_id() { return _partition_id; }

  private:
    MappingTable _mapping_table;
    json _model_config;
    std::string _name;
    onnx::ModelProto _model_proto;
    uint32_t _root_node_id;
    std::map<uint32_t, std::unique_ptr<Operation>> _operation_map;
    std::map<uint32_t, std::unique_ptr<Tensor>> _tensor_map;
    std::vector<Operation*> _executable_layer;
    SimulationConfig _config;
    uint32_t _partition_id = 0;

    /* Number of simulating attention block */
    int nr_skip = 0; // NR_SKIP == 2 * NR_ATTEN
    uint64_t _request_time = 0;   // pico second
    uint64_t _start_time = 0;   // pico second
    bool _started = false;
    bool check_exist_in_exeutable(uint32_t id);
};

#endif