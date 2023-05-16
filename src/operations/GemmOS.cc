#include "GemmOS.h"

#include "../Model.h"

GemmOS::GemmOS(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto)
    : Gemm(config, model, node_proto) {}

/* TODO : Implement this */
void GemmOS::initialize_tiles(MappingTable mapping_table) {

}