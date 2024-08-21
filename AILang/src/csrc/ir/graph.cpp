#include "ir/graph.h"

#include <utility>

#include "ir/node.h"
#include "ir/value.h"

namespace ainl::ir {

Graph::Graph() {
  this->beginBlock = new Block();
  this->endBlock = new Block();
  this->beginBlock->setNext(this->endBlock);
  this->endBlock->setPrev(this->beginBlock);
}

void Graph::insertNodeAtEnd(NodePtr Node) {
  if (this->endBlock->prev != this->beginBlock) {
    ((BlockPtr)(this->endBlock->prev))->insertNodeAtEnd(Node);
  } else {
    // throw AINLError(
    // "Attempting to add a node without first creating a block.");
  }
}

std::string Graph::str() {
  std::string str;
  for (auto bb = (BlockPtr)beginBlock->next; bb->next != nullptr;
       bb = (BlockPtr)bb->next) {
    str.append(std::string(*bb));
  }
  return str;
}

int Graph::GraphParam::FPARAM_COUNT = 0;

Value::ValueKind Graph::getValueKind() const { return Value::ValueKind::Graph; }

Graph::GraphParam::operator std::string() const {
  return type->getName() + " " + getName();
}

Graph::GraphParam::GraphParam(TypePtr type, int idx) : idx(idx) {
  this->type = std::move(type);
  this->prefix = LOCAL_PREFIX;
  this->name = FPARAM_NAME_PREFIX + std::to_string(FPARAM_COUNT++);
}
} // namespace ainl::ir