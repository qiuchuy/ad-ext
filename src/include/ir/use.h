

#ifndef AINL_SRC_INCLUDE_USE_H
#define AINL_SRC_INCLUDE_USE_H

#include "linklist.h"
#include "value.h"

class Use;
class Value;
using UsePtr = Use *;
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

    virtual explicit operator std::string() const;
};

#endif // AINL_SRC_INCLUDE_USE_H