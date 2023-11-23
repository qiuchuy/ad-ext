#include "graph.h"

#include "value.h"
#include <utility>

template <typename NodeType, typename... ARGS>
NodePtr Graph::create(ARGS &&...args) {
    NodePtr Node = new NodeType(std::forward<ARGS>(args)...);
    Node->graph = shared_from_this();
    Node->block = endBlock;
    insertNodeAtEnd(Node);
    return Node;
}

Graph::Graph(std::string name, const TypePtr &inputType)
    : name(std::move(name)) {
    std::vector<ValuePtr> params;
    if (inputType->isTupleType()) {
        std::vector<TypePtr> paramTypes =
            SAFE_TYPE_DOWNCAST(inputType, TupleType)->getTypes();
        for (int idx = 0; (size_t)idx < paramTypes.size(); idx++) {
            auto param = new Graph::Param(paramTypes[idx], idx);
            // param->graph = shared_from_this();
            params.push_back(param);
        }
    } else {
        params.push_back(new Graph::Param(inputType, 0));
    }
    this->beginBlock = new Block(params);
    this->endBlock = new Block();
    this->beginBlock->setNext(endBlock);
    this->endBlock->setPrev(beginBlock);
}

void Graph::insertNodeAtEnd(NodePtr Node) {
    this->endBlock->insertNodeAtEnd(Node);
}

std::string Graph::getName() const { return name; }

std::string Graph::str() {
    std::string str;
    for (auto bb = (BlockPtr)beginBlock->next; bb->next != nullptr;
         bb = (BlockPtr)bb->next) {
        str.append(std::string(*bb)).append(":\n");
    }
    return str;
}

int Graph::Param::FPARAM_COUNT = 0;

Graph::Param::operator std::string() const {
    return type->getName() + " " + getName();
}

Graph::Param::Param(TypePtr type, int idx) : idx(idx) {
    this->type = std::move(type);
    this->prefix = LOCAL_PREFIX;
    this->name = FPARAM_NAME_PREFIX + std::to_string(FPARAM_COUNT++);
}
