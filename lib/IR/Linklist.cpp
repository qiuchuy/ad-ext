#include "ailang/IR/Linklist.h"

namespace ainl::ir {

int ILinkNode::id_num = 0;

bool ILinkNode::hasPrev() const { return prev != nullptr; }

bool ILinkNode::hasNext() const { return next != nullptr; }

void ILinkNode::remove() {
  if (prev != nullptr) {
    prev->next = next;
  }
  if (next != nullptr) {
    next->prev = prev;
  }
}

void ILinkNode::insertAfter(ILinkNode *node) {
  node->prev = this;
  node->next = this->next;
  if (next != nullptr) {
    next->prev = node;
  }
  this->next = node;
}

void ILinkNode::insertBefore(ILinkNode *node) {
  node->next = this;
  node->prev = this->prev;
  if (prev != nullptr) {
    prev->next = node;
  }
  this->prev = node;
}

void ILinkNode::setPrev(ILinkNode *node) { this->prev = node; }

void ILinkNode::setNext(ILinkNode *node) { this->next = node; }

bool ILinkNode::operator==(const ILinkNode &other) const {
  return id == other.id;
}

bool ILinkNode::operator!=(const ILinkNode &other) const {
  return id != other.id;
}

bool ILinkNode::operator<(const ILinkNode &other) const {
  return id < other.id;
}
} // namespace ainl::ir