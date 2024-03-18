

#pragma once

#include "ir/linklist.h"
#include "ir/value.h"

namespace ainl::ir {

class Value;
using ValuePtr = Value *;

class Use : public ILinkNode {
public:
  ValuePtr user{};
  ValuePtr used{};

  int idx = 0;
  static int use_num;
  int id = ++use_num;

public:
  Use() = default;

  Use(ValuePtr user, ValuePtr used, int idx)
      : user(user), used(used), idx(idx){};

  ~Use() override = default;

  friend bool operator==(const Use &first, const Use &second);

  friend bool operator!=(const Use &first, const Use &second);

  friend bool operator<(const Use &first, const Use &second);

  explicit operator std::string() const override;
};
} // namespace ainl::ir