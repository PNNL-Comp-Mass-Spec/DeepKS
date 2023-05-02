"""Contains custom pprint implementations."""

import os, random, copy, typing, pprint, sys, numpy as np


def nice_printer(
    obj,
    max_depth=5,
    max_items=10,
    min_singleton_width=20,
    max_width=int(os.get_terminal_size().columns * 0.8),
    initial_indent=4,
    subsequent_indent=4,
    file=sys.stdout,
):
    global INITIAL_INDENT, SUBSEQUENT_INDENT, MIN_SINGLETON_WIDTH, FILE
    INITIAL_INDENT = initial_indent
    SUBSEQUENT_INDENT = subsequent_indent
    MIN_SINGLETON_WIDTH = min_singleton_width
    FILE = file
    max_str_len = max_width - initial_indent - subsequent_indent
    if isinstance(obj, dict):
        _base_dict_shortener(obj, max_items, max_str_len, initial_indent, is_first=True)
    elif isinstance(obj, typing.Iterable):
        _base_iterable_shortener(obj, max_items, max_str_len, initial_indent)
    else:
        _base_singleton_shortener(obj, max_str_len, "\n")
    print("", file=FILE)


def _base_iterable_shortener(node, max_items, max_str_len, cur_indent):
    print("[", end="", file=FILE)
    for e, v in enumerate(node):
        if isinstance(v, dict):
            _base_dict_shortener(
                v, max_items, max_str_len, cur_indent + SUBSEQUENT_INDENT, addl_end="\n" + " " * (cur_indent)
            )
        elif any([isinstance(v, ty) for ty in [list, tuple, set, frozenset]]):
            _base_iterable_shortener(v, max_items, max_str_len, cur_indent + SUBSEQUENT_INDENT)
        else:
            _base_singleton_shortener(
                v,
                max_str_len,
                ", " if e != len(node) - 1 else "",
                next_len=(next_len := len(str(node[e + 1])) if e < len(node) - 1 else len(str(node[e]))),
                cur_indent=cur_indent,
                indent_exception=e == 0,
            )
            if e in {max_items, len(node) - 1} and next_len > max_str_len and e != 0:
                print(" " * cur_indent, end="", file=FILE)
        if e >= max_items:
            print("... ", end="", file=FILE)
            break
    print("]", end="", file=FILE)


def _base_dict_shortener(node, max_items, max_str_len, cur_indent, addl_end="", is_first=False):
    if not is_first:
        print("", file=FILE)
    print(" " * (cur_indent - 1) + "{", end="", file=FILE)
    for e, (k, v) in enumerate(node.items()):
        if e != 0:
            print(" " * cur_indent, end="", file=FILE)
        _base_singleton_shortener(k, max_str_len // 4, ": ")
        if isinstance(v, dict):
            _base_dict_shortener(v, max_items, max_str_len, cur_indent + SUBSEQUENT_INDENT)
        elif any([isinstance(v, ty) for ty in [list, tuple, set, frozenset]]):
            _base_iterable_shortener(v, max_items, max_str_len, cur_indent + SUBSEQUENT_INDENT)
        else:
            _base_singleton_shortener(v, max_str_len, "")
        if e >= max_items and not is_first:
            print("... ", end="", file=FILE)
            break
        print("", file=FILE)
    print(" " * (cur_indent - 1) + "}", end="", file=FILE)
    print(addl_end, end="", file=FILE)


def _base_singleton_shortener(node, max_str_len, sep, indent="", next_len=0.0, cur_indent=0, indent_exception=False):
    old_node_str = str(node)
    new_str = old_node_str[: max_str_len - cur_indent] + "... " if len(old_node_str) > max_str_len else old_node_str
    do_indent = False
    if next_len > max_str_len - cur_indent and len(old_node_str) > max_str_len:
        sep = sep + "\n"
        do_indent = not indent_exception
    if do_indent:
        indent += " " * cur_indent
    print(indent + new_str, end=sep, file=FILE)


if __name__ == "__main__":
    random.seed(42)
    # test_obj = {
    #     "a": [1, 2, 3, 4, 5, 6, {"A": "B", "CC": list(range(200))}],
    #     "b": {
    #         1: [chr(x) for x in [random.choice(range(65, 65 + 26)) for _ in range(1000)]],
    #         2: [chr(x) for x in [random.choice(range(65, 65 + 26)) for _ in range(1000)]],
    #         3: [chr(x) for x in [random.choice(range(65, 65 + 26)) for _ in range(1000)]],
    #         4: [chr(x) for x in [random.choice(range(97, 97 + 26)) for _ in range(1000)]],
    #         "".join([chr(x) for x in [random.choice(range(48, 48 + 10)) for _ in range(1000)]]): ["".join([
    #             chr(x) for x in [random.choice(range(48, 48 + 10)) for _ in range(1000)]
    #         ]) for _ in range(20)],
    #     },
    # }
    test_obj = {"a": {"A": {"AA": {"AAA": [1, 2, 3, 4]}}}, "b": {}, "c": [1, {2: 3}, {4: {5: 6}}]}

    nice_printer(test_obj)
