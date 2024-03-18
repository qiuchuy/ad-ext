from typing import Callable
import ailang

# Copyright © 2023 Apple Inc.

from collections import defaultdict


def tree_map(fn, tree, *rest, is_leaf=None):
    if is_leaf is not None and is_leaf(tree):
        return fn(tree, *rest)
    elif isinstance(tree, (list, tuple)):
        TreeType = type(tree)
        return TreeType(
            tree_map(fn, child, *(r[i] for r in rest), is_leaf=is_leaf)
            for i, child in enumerate(tree)
        )
    elif isinstance(tree, dict):
        return {
            k: tree_map(fn, child, *(r[k] for r in rest), is_leaf=is_leaf)
            for k, child in tree.items()
        }
    else:
        return fn(tree, *rest)


def tree_flatten(tree, prefix="", is_leaf=None):
    """Flattens a python tree to a list of key, value tuples."""
    flat_tree = []

    if is_leaf is None or not is_leaf(tree):
        if isinstance(tree, (list, tuple)):
            for i, t in enumerate(tree):
                flat_tree.extend(tree_flatten(t, f"{prefix}.{i}", is_leaf))
            return flat_tree
        if isinstance(tree, dict):
            for k, t in tree.items():
                flat_tree.extend(tree_flatten(t, f"{prefix}.{k}", is_leaf))
            return flat_tree

    return [(prefix[1:], tree)]


def tree_unflatten(tree):
    """Recreate a python tree from its flat representation."""
    if len(tree) == 1 and tree[0][0] == "":
        return tree[0][1]

    try:
        int(tree[0][0].split(".", maxsplit=1)[0])
        is_list = True
    except ValueError:
        is_list = False

    # collect children
    children = defaultdict(list)
    for key, value in tree:
        current_idx, *next_idx = key.split(".", maxsplit=1)
        next_idx = "" if not next_idx else next_idx[0]
        children[current_idx].append((next_idx, value))

    # recursively map them to the original container
    if is_list:
        keys = sorted((int(idx), idx) for idx in children.keys())
        l = []
        for i, k in keys:
            # if i <= len(l), no {} will be appended.
            l.extend([{} for _ in range(i - len(l))])
            l.append(tree_unflatten(children[k]))
        return l
    else:
        return {k: tree_unflatten(v) for k, v in children.items()}
