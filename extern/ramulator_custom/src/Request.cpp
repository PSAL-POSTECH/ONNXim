#include "Request.h"

namespace ram {

Request::Request() {}

Request::Request(Type Type, uint64_t Addr, std::vector<int> AddrVec,
                 function<void(const Request&)> &cb)
    : type(Type),
      is_first_command(true),
      addr(Addr),
      addr_vec(AddrVec),
      coreid(0),
      arrive(0),
      depart(0),
      callback(cb) {}

Request::Request(Type Type, uint64_t Addr, std::vector<int> AddrVec,
                 function<void(const Request&)> &cb, void* original_req)
    : type(Type),
      is_first_command(true),
      addr(Addr),
      addr_vec(AddrVec),
      coreid(0),
      arrive(0),
      depart(0),
      callback(cb),
      orignal_request(original_req) {}

Request::Request(Type Type, uint64_t Addr, std::vector<int> AddrVec,
                 function<void(const Request&)> &cb, int vid)
    : type(Type),
      is_first_command(true),
      addr(Addr),
      addr_vec(AddrVec),
      coreid(0),
      arrive(0),
      depart(0),
      vid(vid),
      callback(cb) {}

Request::Request(std::vector<int> addr_vec, Type type,
                 function<void(Request&)> cb) 
    : type(type),
      is_first_command(true),
      addr(-1),
      BaseAddr(-1),
      addr_vec(addr_vec),
      coreid(0),
      arrive(0),
      depart(0),
      callback(cb) {}
      
Request::Request(std::vector<int> addr_vec, Type type,
                 function<void(Request&)> cb, void* original_req) 
    : type(type),
      is_first_command(true),
      addr(-1),
      BaseAddr(-1),
      addr_vec(addr_vec),
      coreid(0),
      arrive(0),
      depart(0),
      callback(cb),
      orignal_request(original_req) {}

Request::Request(Type Type, uint64_t BaseAddr, uint64_t Addr, 
                 std::vector<int> AddrVec, function<void(const Request&)> &cb)
    : type(Type),
      is_first_command(true),
      addr(Addr),
      BaseAddr(BaseAddr),
      addr_vec(AddrVec),
      coreid(0),
      arrive(0),
      depart(0),
      callback(cb) {}

bool Request::isRead() const {
  return type == Type::READ;
}
bool Request::isWrite() const {
  return type == Type::WRITE;
}
int Request::getChannelID() const {
  return addr_vec[0];
}

} // end namespace

