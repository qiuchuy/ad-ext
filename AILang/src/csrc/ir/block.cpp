#include "ir/block.h"

namespace ainl::ir {
int Block::blockCount = 0;

Block::Block() {
  paramNode = nullptr;
  returnNode = nullptr;
  beginNode = new Node();
  endNode = new Node();
  beginNode->setNext(endNode);
  endNode->setPrev(beginNode);
  beginBlock = endBlock = nullptr;
}

Block::Block(int idx) {
  paramNode = nullptr;
  returnNode = nullptr;
  beginNode = new Node();
  endNode = new Node();
  beginNode->setNext(endNode);
  endNode->setPrev(beginNode);
  beginBlock = endBlock = nullptr;
  label = "b" + std::to_string(idx);
}

std::vector<ValuePtr> Block::getParams() { return paramNode->getParams(); }
void Block::insertNodeAtHead(NodePtr Node) { beginNode->insertAfter(Node); }
void Block::insertNodeAtEnd(NodePtr Node) { endNode->insertBefore(Node); }

Block::operator std::string() const {
  std::string str;
  str.append("{\n");
  // str.append(label + ":\n");
  for (auto node = (NodePtr)beginNode->next; node->next != nullptr;
       node = (NodePtr)node->next) {
    str.append("\t");
    str.append(std::string(*node));
    str.append("\n");
  }
  str.append("}\n");
  return str;
}
} // namespace ainl::ir