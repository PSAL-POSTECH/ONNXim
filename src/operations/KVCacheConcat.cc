#include "KVCacheConcat.h"

KVCacheConcat::KVCacheConcat(SimulationConfig config, Model* model,
                             onnx::NodeProto& node_proto)
    : Operation(config, model, node_proto) {}

KVCacheConcat::KVCacheConcat(const KVCacheConcat& src)
    : Operation(src) {}

KVCacheConcat::KVCacheConcat(SimulationConfig config, Model* model,
                             std::string name,
                             std::map<std::string, std::string>& attributes)
    : Operation(config, model, name, attributes) {}

void KVCacheConcat::initialize_tiles(MappingTable& mapping_table) {
  //TODO:implemt 
}