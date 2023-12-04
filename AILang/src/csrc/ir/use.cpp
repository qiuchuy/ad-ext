#include "use.h"

#include <cassert>
#include <sstream>
#include <string>

int Use::use_num = 0;

bool operator==(const Use &first, const Use &second) {
    return first.id == second.id;
}

bool operator!=(const Use &first, const Use &second) {
    return first.id != second.id;
}

bool operator<(const Use &first, const Use &second) {
    return first.id < second.id;
}

Use::operator std::string() const {
    std::stringstream ss;
    assert(this->used != nullptr && this->user != nullptr);
    ss << used->getName() << "@[" << std::string(*user) << "]";
    return ss.str();
}