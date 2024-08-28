#include "GemmOS.h"

#include "../Model.h"

GemmOS::GemmOS(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto, uint32_t target_core)
    : Gemm(config, model, node_proto, target_core) {}

/* TODO : Implement this */
void GemmOS::initialize_tiles(MappingTable& mapping_table) {

}