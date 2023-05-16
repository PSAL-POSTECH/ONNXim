#include "Core.h"

class SystolicWS : public Core {
 public:
  SystolicWS(uint32_t id, SimulationConfig config);
  virtual void cycle() override;
  virtual void print_stats() override;

 protected:
  virtual cycle_type get_inst_compute_cycles(Instruction inst) override;
  uint32_t _stat_systolic_inst_issue_count = 0;
  uint32_t _stat_systolic_preload_issue_count = 0;
};