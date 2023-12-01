#ifndef AINL_SRC_INCLUDE_LINKLIST_H
#define AINL_SRC_INCLUDE_LINKLIST_H

#include <string>

class ILinkNode {
  public:
    ILinkNode *prev{};
    ILinkNode *next{};
    static int id_num;
    int id = id_num++;

  public:
    ILinkNode() = default;

    virtual ~ILinkNode() = default;

    bool hasPrev() const;

    bool hasNext() const;

    virtual void remove();

    void setPrev(ILinkNode *prev);

    void setNext(ILinkNode *next);

    void insertAfter(ILinkNode *node);
    void insertBefore(ILinkNode *node);
    virtual explicit operator std::string() const { return ""; }
    virtual bool operator==(const ILinkNode &other) const;

    virtual bool operator!=(const ILinkNode &other) const;

    bool operator<(const ILinkNode &other) const;
};

#endif // AINL_SRC_INCLUDE_LINKLIST_H