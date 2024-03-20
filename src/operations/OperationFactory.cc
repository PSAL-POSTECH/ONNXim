#include "OperationFactory.h"

#include "AdaptiveAvgPool.h"
#include "Conv.h"
#include "ConvOS.h"
#include "ConvWS.h"
#include "Flatten.h"
#include "Gemm.h"
#include "GemmWS.h"
#include "GlobalAvgPool.h"
#include "Operation.h"
#include "Attention.h"
#include "Cast.h"
#include "EmbedLayerNorm.h"
#include "SkipLayerNorm.h"
// #include "MatMul.h"
#include "MaxPool.h"

SimulationConfig OperationFactory::_config = SimulationConfig();

void OperationFactory::initialize(SimulationConfig config) { _config = config; }

std::unique_ptr<Operation> OperationFactory::create_operation(
    Model* model, onnx::NodeProto& node_proto) {
  if (node_proto.op_type() == "Conv" || node_proto.op_type() == "FusedConv") {
    if (_config.core_type == CoreType::SYSTOLIC_OS)
      return std::make_unique<ConvOS>(_config, model, node_proto);
    else if (_config.core_type == CoreType::SYSTOLIC_WS)
      return std::make_unique<ConvWS>(_config, model, node_proto);
  } else if (node_proto.op_type() == "Gemm" ||
             node_proto.op_type() == "FusedGemm") {
    if (_config.core_type == CoreType::SYSTOLIC_WS)
      return std::make_unique<GemmWS>(_config, model, node_proto);
  } else if (node_proto.op_type() == "MatMul") {
    return std::make_unique<GemmWS>(_config, model, node_proto, false);
  } else if (node_proto.op_type() == "MaxPool") {
    return std::make_unique<MaxPool>(_config, model, node_proto);
  } else if (node_proto.op_type() == "GlobalAveragePool") {
    return std::make_unique<GlobalAvgPool>(_config, model, node_proto);
  } else if (node_proto.op_type() == "AdaptiveAveragePool" ||
             node_proto.op_type() == "AveragePool") {
    return std::make_unique<AdaptiveAvgPool>(_config, model, node_proto);
  } else if (node_proto.op_type() == "Flatten") {
    return std::make_unique<Flatten>(_config, model, node_proto);
  } else if (node_proto.op_type() == "Attention") {
    return std::make_unique<Attention>(_config, model, node_proto);
  } else if (node_proto.op_type() == "Cast") {
    return std::make_unique<Cast>(_config, model, node_proto);
  } else if (node_proto.op_type() == "EmbedLayerNormalization") {
    return std::make_unique<EmbedLayerNorm>(_config, model, node_proto);
  } else if (node_proto.op_type() == "SkipLayerNormalization") {
    return std::make_unique<SkipLayerNorm>(_config, model, node_proto);
  }
  spdlog::warn("Node Proto optype \"{}\" returned nullptr",
               node_proto.op_type().c_str());
  return nullptr;
}

std::unique_ptr<Operation> OperationFactory::copy_operation(Operation* op) {
  if (op->get_optype() == "Conv" || op->get_optype() == "FusedConv") {
    if (_config.core_type == CoreType::SYSTOLIC_OS)
      return std::make_unique<ConvOS>(*dynamic_cast<ConvOS*>(op));
    else if (_config.core_type == CoreType::SYSTOLIC_WS)
      return std::make_unique<ConvWS>(*dynamic_cast<ConvWS*>(op));
  } else if (op->get_optype() == "Gemm" || op->get_optype() == "FusedGemm") {
    if (_config.core_type == CoreType::SYSTOLIC_WS)
      return std::make_unique<GemmWS>(*dynamic_cast<GemmWS*>(op));
  } else if (op->get_optype() == "MatMul") {
      return std::make_unique<GemmWS>(*dynamic_cast<GemmWS*>(op));
  } else if (op->get_optype() == "MaxPool") {
    return std::make_unique<MaxPool>(*dynamic_cast<MaxPool*>(op));
  } else if (op->get_optype() == "AdaptiveAveragePool" ||
             op->get_optype() == "AveragePool") {
    return std::make_unique<AdaptiveAvgPool>(
        *dynamic_cast<AdaptiveAvgPool*>(op));
  } else if (op->get_optype() == "GlobalAveragePool") {
    return std::make_unique<GlobalAvgPool>(*dynamic_cast<GlobalAvgPool*>(op));
  } else if (op->get_optype() == "Flatten") {
    return std::make_unique<Flatten>(*dynamic_cast<Flatten*>(op));
  } else if (op->get_optype() == "Attention") {
    return std::make_unique<Attention>(*dynamic_cast<Attention*>(op));
  } else if (op->get_optype() == "Cast") {
    return std::make_unique<Cast>(*dynamic_cast<Cast*>(op));
  } else if (op->get_optype() == "EmbedLayerNormalization") {
    return std::make_unique<EmbedLayerNorm>(*dynamic_cast<EmbedLayerNorm*>(op));
  } else if (op->get_optype() == "SkipLayerNormalization") {
    return std::make_unique<SkipLayerNorm>(*dynamic_cast<SkipLayerNorm*>(op));
  }
  spdlog::warn("Node Proto optype \"{}\" returned nullptr", op->get_optype());
  return nullptr;
}
