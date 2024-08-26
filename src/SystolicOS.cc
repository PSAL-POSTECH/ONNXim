#include "SystolicOS.h"

SystolicOS::SystolicOS(uint32_t id, SimulationConfig config)
    : Core(id, config) {}

void SystolicOS::cycle() {
  // Todo: Impement this;
  assert(0);
}

cycle_type SystolicOS::get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) {
  return _config.core_config[_id].core_height + _config.core_config[_id].core_width - 2 + inst->size;
}