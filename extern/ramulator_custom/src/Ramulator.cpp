#include "Ramulator.hpp"
#include "Memory.h"
#include "MemoryFactory.h"
#include "DDR4.h"
#include "HBM.h"
#include "Request.h"

namespace ram {
  // TODO: init outputpendingqueue
Ramulator::Ramulator(const std::string ConfigFilePath, uint32_t num_core, bool is_pim) 
    : MemBase(createMemory(ConfigFilePath, num_core)), is_pim(is_pim) {
  for (int ch = 0; ch < MemBase->get_num_channels(); ++ch) {
    OutputPendingQueues.push_back(OutputPendingQueue(64));
  }
  Callbacks[false] = [&](const ram::Request& Req) {
    int CtrlID = Req.getChannelID();
    // TODO: check pending queue reservation logic
    OutputPendingQueues[CtrlID].push(Req.orignal_request);
  };
  Callbacks[true] = [&](const ram::Request& Req) {
    int CtrlID = Req.getChannelID();
    // // TODO: check pending queue reservation logic
    OutputPendingQueues[CtrlID].push(Req.orignal_request);
  };

  if (is_pim) {
    int hot_vid = -1;
    int in_degrees = -1;
    int total_vid = 0;
  }
  Stat::statlist.output("./ramulator.stats");
}

void Ramulator::tick() {
  MemBase->tick();
}

bool Ramulator::isAvailable(int CtrlID, uint64_t Addr, bool IsWrite) const {
  std::vector<int> MemAddr = MemBase->decode_mem_addr(Addr);
  assert(CtrlID == MemAddr[0]);
  return  OutputPendingQueues[CtrlID].isAvailable(1) && !MemBase->is_full(CtrlID, IsWrite);
}

bool Ramulator::isAvailable(uint64_t Addr, bool IsWrite) const {
  // TODO: need to avoid decoding memory addr whenever `isAvailable` is called
  std::vector<int> MemAddr = MemBase->decode_mem_addr(Addr);
  uint32_t CtrlID = MemAddr[0];
    
  bool result = OutputPendingQueues[CtrlID].isAvailable(1) && !MemBase->is_full(CtrlID, IsWrite);
  
  return result;
}


void Ramulator::push(int CtrlID, uint64_t Addr, bool IsWrite, uint32_t core_id, void* orignal_req) {
  std::vector<int> MemAddr = MemBase->decode_mem_addr(Addr);
  //Ensure CtrlID match with decoded address
  assert(CtrlID == MemAddr[0]); 
  if (IsWrite) {
    Request req(Request::Type::WRITE, Addr, MemAddr, Callbacks[IsWrite], orignal_req);
    req.coreid = core_id;
    bool isSent = MemBase->send(req);
    assert(isSent);
  } else {
    Request req(Request::Type::READ, Addr, MemAddr, Callbacks[IsWrite], orignal_req);
    req.coreid = core_id;
    bool isSent = MemBase->send(req);
    assert(isSent);
  }

  OutputPendingQueues[CtrlID].reserve();
}

void Ramulator::push(uint64_t Addr, bool IsWrite, uint32_t core_id, void* original_req) {
  std::vector<int> MemAddr = MemBase->decode_mem_addr(Addr);
  const int CtrlID = MemAddr[0];
  // TODO: vid check here
  if (IsWrite) {
    Request req(Request::Type::WRITE, Addr, MemAddr, Callbacks[IsWrite], original_req);
    req.coreid = core_id;
    bool isSent = MemBase->send(req);
    assert(isSent);
  } else {
    Request req(Request::Type::READ, Addr, MemAddr, Callbacks[IsWrite], original_req);
    req.coreid = core_id;
    bool isSent = MemBase->send(req);
    assert(isSent);
  }

  OutputPendingQueues[CtrlID].reserve();
}

bool Ramulator::isEmpty(int CtrlID) const {
  return OutputPendingQueues[CtrlID].isEmpty();
}
const void* Ramulator::top(int CtrlID) const {
  return OutputPendingQueues[CtrlID].top();
}
void Ramulator::pop(int CtrlID) {
  OutputPendingQueues[CtrlID].pop();
}

int Ramulator::getAtomicBytes() const {
  return MemBase->get_transaction_bytes();
}

int Ramulator::getNumChannels() const {
  return MemBase->get_num_channels();
}

int Ramulator::getChannel(uint64_t Addr) const {
  std::vector<int> MemAddr = MemBase->decode_mem_addr(Addr);
  return MemAddr[0];
}

void Ramulator::print_stats() {
  MemBase->finish();
  Stat::statlist.printall();
}

std::unique_ptr<MemoryBase> 
Ramulator::createMemory(const std::string ConfigFilePath, uint32_t num_core) {
  RamulatorConfig Config(ConfigFilePath);
  Config.set_core_num(num_core);
  std::string MemType = Config["standard"];
  if (MemType == "DDR4") {
    return MemoryFactory<DDR4>::create(Config, 32);
  } else if (MemType == "HBM") {
    return MemoryFactory<HBM>::create(Config, 32);
  } else {
    assert(false);
    return nullptr;
  }
}
Ramulator::OutputPendingQueue::OutputPendingQueue(int Size)
    : Size(Size),
      NumReserved(0) {}

bool Ramulator::OutputPendingQueue::isAvailable() const {
  return NumReserved + PendingQueue.size() < Size;
}

bool Ramulator::OutputPendingQueue::isAvailable(uint32_t count) const {
  return NumReserved + PendingQueue.size() + count - 1 < Size;
}

void Ramulator::OutputPendingQueue::reserve() {
  assert(NumReserved < Size);
  NumReserved++;
}

void Ramulator::OutputPendingQueue::push(void* Addr) {
  PendingQueue.push(Addr);
  assert(NumReserved > 0);
  NumReserved--;
}

bool Ramulator::OutputPendingQueue::isEmpty() const {
  return PendingQueue.empty();
}

void Ramulator::OutputPendingQueue::pop() {
  PendingQueue.pop();
}
const void* Ramulator::OutputPendingQueue::top() const {
  return PendingQueue.front();
}

Ramulator::~Ramulator() = default;

}
