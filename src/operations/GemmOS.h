#pragma once
#include "Gemm.h"

class GemmOS : public Gemm {
  public:
    GemmOS(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    void initialize_tiles(MappingTable& mapping_table) override;
  private:
};