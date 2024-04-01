#include "Core.h"

class SystolicOS : public Core {
  public:
  SystolicOS(uint32_t id, SimulationConfig config);
  virtual void cycle() override;
  protected:
  virtual cycle_type get_inst_compute_cycles(std::unique_ptr<Instruction>& inst);
};