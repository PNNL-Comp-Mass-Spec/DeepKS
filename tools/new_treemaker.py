import json
import os
import re
from ..config.join_first import join_first

class FDNode():
    def __init__(self, name: str, full_path: list[str], description: str):
        self.name = name
        self.full_path = full_path
        self.description = description
        self.level = len(description) - 1

class FileNode(FDNode):
    def __init__(self, name: str, full_path: list[str], description: str):
        super().__init__(name, full_path, description)

class DirNode(FDNode):
    def __init__(self, name: str, full_path: list[str], description: str, value: list[FDNode]):
        super().__init__(name, full_path, description)

name_to_node: dict[str, tuple[str, list[str], list[str]]] = {}

def main(root=join_first("", 1, __file__), excluded_file_name = [], included_file_name = [], excluded_dir = [], included_dir = []):
    the_walk = list(os.walk(root))
    for node in the_walk:
        name_to_node[node[0]] = node
    return walk_to_dict(the_walk[0], excluded_file_name, included_file_name, excluded_dir, included_dir)


def walk_to_dict_helper(node: tuple[str, list[str], list[str]], excluded_file_name, included_file_name, excluded_dir, included_dir):
    partial_dict = {}
    partial_dict[node[0]] = {}
    for child_dir in node[1]:
        child_dir = os.path.join(node[0], child_dir)
        do_continue = False
        for ptn in excluded_dir:
            if re.search(ptn, child_dir):
                do_continue = True
        for ptn in included_dir:
            if re.search(ptn, child_dir):
                do_continue = False
        if do_continue:
            continue
        wtdh = walk_to_dict_helper(name_to_node[child_dir], excluded_file_name, included_file_name, excluded_dir, included_dir)
        for k, v in wtdh.items():
            partial_dict[node[0]][(k, "Dir Desc")] = v
    for child_file in node[2]:
        do_continue = False
        for ptn in excluded_file_name:
            if re.search(ptn, os.path.join(node[0], child_file)):
                do_continue = True
        for ptn in included_file_name:
            if re.search(ptn, os.path.join(node[0], child_file)):
                do_continue = False
        if do_continue:
            continue
        partial_dict[node[0]][(child_file.split("/")[-1], "File Desc")] = None
        
    return dict({k.split("/")[-1]: v for k, v in partial_dict.items()})


def walk_to_dict(root_node: tuple[str, list[str], list[str]], excluded_file_name, included_file_name, excluded_dir, included_dir):
    # root_node = ((root_node[0], "Root Desc"), root_node[1], root_node[2])
    return walk_to_dict_helper(root_node, excluded_file_name, included_file_name, excluded_dir, included_dir)


def traverse_nested(nested: dict):
    return traverse_nested_helper(nested, 0, set())

VBAR = "│"
HBAR = "─ "
VBARHBAR = "├ "
ENDBAR = "└ "
DIRSYMB = "• "
DESCSYMB = " ▷ "
def traverse_nested_helper(nested: dict, level: int, no_vert_levels: set[int]) -> str:
    build_up = ""
    def sort_key(x: tuple) -> tuple:
        # print(x)
        if isinstance(x[0], str):
            x = ((x[0], "No Desc"), x[1])
        x = (x[0][0], x[1])
        if x[1] is None:
            return ("", x[0])
        else:
            return (x[0], x[0])
        # return x
        
    for i, (k, v) in enumerate(ll := sorted(list(nested.items()), key = sort_key)):
        spaces = ""
        for l in range(1, level):
            if l in no_vert_levels:
                spaces += "  "
            else:
                spaces += f"{VBAR} "
        
        if level == 0:
            syb = ""
        elif i + 1 == len(ll):
            syb = f"{ENDBAR}"
        else:
            syb = f"{VBARHBAR}"
        
        if v is None:
            if isinstance(k, str):
                k = (k, "No Desc")
            build_up += spaces + syb + k[0] + DESCSYMB + k[1] + "\n"
        else:
            add_on_set: set[int]
            if i + 1 == len(ll):
                add_on_set = set.union(no_vert_levels, {level})
            else:
                add_on_set = no_vert_levels
            if isinstance(k, str):
                k = (k, "No Desc")
            build_up += spaces + syb + DIRSYMB + k[0] + DESCSYMB + k[1] + "\n" + traverse_nested_helper(v, level + 1, add_on_set)
    return build_up
    


if __name__ == "__main__": # pragma: no cover
    excluded_both = [r"__pycache__", r"__init__", r"vscode", r"\.git", r"docs/api_pydoctor_docs/[^\.]+", r"out/[^\.]+"]
    excluded_file_name=[r"\.pyc", r"\.DS_Store", r"\.gitig-*", r"tree.txt"] + excluded_both
    included_file_name=[r'docs/api_pydoctor_docs/index.html', r"results_2023-05-03@19`00`32.0@-04`00.json", r"results_2023-05-03@19`00`34.4@-04`00.csv", r"results_2023-05-03@19`00`36.6@-04`00.sqlite"]
    excluded_dir = excluded_both
    included_dir = []

    tree_trav = main(join_first("", 1, __file__), excluded_file_name, included_file_name, excluded_dir, included_dir)
    with open("tree_data.json", "w") as f:
        json.dump(tree_trav, f) 
    # pprint(tree_trav)
    pretty = traverse_nested(tree_trav)
    print(pretty)
