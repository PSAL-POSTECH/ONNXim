#ifndef __REQUEST_H
#define __REQUEST_H

#include <vector>
#include <functional>
#include <cstdint>

using namespace std;

namespace ram {
class Request {
public:
  enum class Type {
    READ, WRITE, PIM_WRITE, REFRESH, POWERDOWN, SELFREFRESH, EXTENSION, MAX
  };
  Type type;
  bool is_first_command;
  uint64_t addr;
  uint64_t BaseAddr;
  //int HandlerID;

  vector<int> addr_vec;
  // specify which node this request sent from
  int coreid;       // to remove compile errors

  uint64_t arrive;
  uint64_t depart;

  int vid = -1;
  void* orignal_request;
  function<void(Request&)> callback; // call back with more info

  bool isRead() const;
  bool isWrite() const;
  int getChannelID() const;

  // Used to generate refresh request
  Request();
  Request(std::vector<int> addr_vec, Type type, function<void(Request&)> cb);
  Request(std::vector<int> addr_vec, Type type, function<void(Request&)> cb, void* original_req);
  Request(Type type, uint64_t Addr, 
          std::vector<int> AddrVec, function<void(const Request&)> &cb);
  Request(Type type, uint64_t Addr, 
          std::vector<int> AddrVec, function<void(const Request&)> &cb, void* orignal_req);
  Request(Type type, uint64_t Addr, 
          std::vector<int> AddrVec, function<void(const Request&)> &cb, int vid);
  Request(Type type, uint64_t BaseAddr, uint64_t Addr, 
          std::vector<int> AddrVec, function<void(const Request&)> &cb);
};

} /*namespace ram*/

#endif /*__REQUEST_H*/

