#pragma once
#include "../Common.h"
#include "Operation.h"

class Model; 

class OperationFactory {
  public:
    static void initialize(SimulationConfig config);
    static std::unique_ptr<Operation> create_operation(Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    static std::unique_ptr<Operation> copy_operation(Operation* op);

  private:
    static SimulationConfig _config;
};