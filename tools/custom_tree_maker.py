from __future__ import annotations
import os, json, re, pprint, time, pathlib
from sys import argv
from typing import Union

DUMP = True
RESTORE = "-r" in argv
DRY = "--dry-run" in argv

TOP_DIR = f"{pathlib.Path(__file__).parent.resolve()}/../"

# TODO: Make lines 90 degrees if end of inner list

def main():
    os.chdir(TOP_DIR)
    ignore_patterns = ["*.pyc", "__pycache__", "__init__.py", ".git", ".DS_Store", ".vscode", ".gitig-*", "tree.txt"]
    ignore_patterns = [f" -I \"{p}\"" for p in ignore_patterns]
    ignore_patterns = "".join(ignore_patterns)
    cmd = f'tree -a -o {TOP_DIR}docs/tree.txt {ignore_patterns}'

    if DRY:
        print("Would have run: " + cmd)
        exit(0)
    exitcode = os.system(cmd)
    if exitcode:
        print(f"`tree` didn't run correctly (exit code {exitcode})")
        exit(1)
    if (os.path.exists(f"{TOP_DIR}docs/tree.html") and len(argv) == 1) or (len(argv) > 1 and "-f" not in argv):
        print("Use -f to forcefully overwrite tree.html")
        exit(1)
    description = {}
    if not RESTORE:
        with open(argv[argv.index('-d') + 1], "r") as d:
            description = d.readlines()

    with open(f"{TOP_DIR}docs/tree.txt", "r") as t:
        lines = t.readlines()[:-2]
    os.unlink(f"{TOP_DIR}docs/tree.txt")
    while os.path.exists(f"{TOP_DIR}docs/tree.txt"):
        time.sleep(0.01)

    new_lines = []
    tree_repr = graph_from_text(lines).get_pre_order_trav()
    

    lines_restore = [None]*len(lines)
    lines_restored = None
    if RESTORE:
        lines_restore = lines.copy()
        lines_restored = []
        with open(f"{TOP_DIR}saved_dict_repr_file.json", "r") as f:
            tree_description_path_dict = {eval(k): v for k, v in json.load(f).items()}
    else:
        assert len(description) != 0; "RESTORE is not True, but there is no description information provided."
        tree_description_path_dict = graph_from_text(description).get_dict_repr()
    
    if DUMP:
        with open(f"{TOP_DIR}saved_dict_repr_file.json", "w") as f:
            json.dump({str(p): v for p, v in tree_description_path_dict.items()}, f, indent=4, sort_keys=True)
    
    bg_class = 'odd'
    
    for i, (orig_line, node, to_restore_line) in enumerate(zip(lines, tree_repr, lines_restore)):
        node_inner = node
        path_to_node = [node_inner.text]
        while node_inner.parent is not None:
            path_to_node.append(node_inner.parent.text)
            node_inner = node_inner.parent
        path_to_node = tuple(path_to_node)
        desc = None
        if path_to_node in tree_description_path_dict:
            desc = tree_description_path_dict[path_to_node]
        else:
            print(f"{'/'.join(reversed(path_to_node))} found in file tree, but there was no corresponding entry in tree_description.txt.")
            # print(f"{path_to_node=}")
        to_restore_line = (to_restore_line if desc is None else to_restore_line.replace("\n", "") + ";" + desc) if to_restore_line is not None else None
        if desc is not None:
            desc = re.sub(r"¡(.*)¡", "<span class='warn'><i>\\1</i></span>", desc)
            bg_class = 'odd' if bg_class == 'even' else 'even'
            orig_line = orig_line.replace("\n", "").replace(" ", " ")
            try:
                bolded = '' if not node.is_directory else ' class=\"dir\"'
                new_line = (
                    "<code"
                    f" class='no-col'>{''.join(re.findall(r'[│─ ├└]', orig_line))}</code><code{bolded}>{re.findall(r'─ (.*)', orig_line)[0] if orig_line != '.' else '.'}</code>"
                )
                
                if i != 0 and tree_repr[i - 1].depth > node.depth:
                    # new_lines[-1] = new_lines[-1].replace("├", "└")
                    pass
            except Exception as e:
                print(e)
                print(f"{orig_line=}{desc=}{node=}")
                exit(1)
            new_lines.append(new_line)
            if desc != "\n":
                new_lines[-1] += " ▷  " + desc.replace("\n", "")
            else:
                new_lines[-1] += desc
            
            new_lines[-1] = f"<div style='display:table-row' class={bg_class}>{new_lines[-1]}</div>".replace("└", "├")
        if RESTORE:
            if to_restore_line is not None and lines_restored is not None:
                lines_restored.append(to_restore_line)
    if RESTORE:
        assert isinstance(lines_restored, list)
        with open(f"{TOP_DIR}docs/tree_description.txt", "w") as d:
            d.write("".join(lines_restored))

    with open(f"{TOP_DIR}docs/tree_template.html") as tt:
        template = tt.read()
    with open(f"{TOP_DIR}docs/tree.html", "w") as t:
        # new_lines[-1] = new_lines[-1].replace("├", "└")
        temp_lines = "\n".join(new_lines)
        final_lines = re.sub("        PUT FILE DIRS HERE", temp_lines, template)
        t.write(final_lines)


class GraphFromTextNode:
    def __init__(self, text: str, depth: int, index_in_list: int, description:Union[str, None]=""):
        self.children: list[GraphFromTextNode] = []
        self.parent: Union[GraphFromTextNode, None] = None
        self.is_directory: bool = False
        self.text = text
        self.text = re.sub(r"[\n│\s└├──]", "", self.text)
        self.description = description
        self.depth = depth
        self.index_in_list = index_in_list

    def __str__(self):
        return f"(({self.text}|{self.description.strip() if self.description is not None else None}))"

    def __repr__(self):
        return self.__str__()

    def add_child(self, child: GraphFromTextNode):
        self.children.append(child)


class GraphFromText:
    def __init__(self, root: GraphFromTextNode):
        self.root = root

    def __str__(self):
        self._repr_str_build_up = []
        self._str_helper(self.root)
        ret = "\n".join(self._repr_str_build_up)
        self._repr_str_build_up = []
        return ret

    def _str_helper(self, node):
        self._repr_str_build_up.append("  "*node.depth + node.__str__())
        if len(node.children) > 0:
            for child in node.children:
                self._str_helper(child)

    def __repr__(self):
        return self.__str__()

    def get_pre_order_trav(self):
        self.res = []
        self._get_pre_order_trav_helper(self.root)
        ret: list[GraphFromTextNode] = self.res
        self.res = []
        return ret

    def _get_pre_order_trav_helper(self, node: GraphFromTextNode):
        self.res.append(node)
        if len(node.children) > 0:
            for child in node.children:
                self._get_pre_order_trav_helper(child)
        else:
            return node

    def get_dict_repr(self):
        self.res_d = {}
        self._get_dict_repr_helper(self.root)
        ret = self.res_d
        self.res_d = {}
        return ret

    def _get_dict_repr_helper(self, node):
        node_inner = node
        abs_path_to_node = [node_inner.text]
        while node_inner.parent is not None:
            abs_path_to_node.append(node_inner.parent.text)
            node_inner = node_inner.parent

        self.res_d[tuple(abs_path_to_node)] = node.description

        if len(node.children) > 0:
            for child in node.children:
                self._get_dict_repr_helper(child)
        else:
            return node


def graph_from_text(lines) -> GraphFromText:
    def get_depth(string:str) -> int:
        dont_match = set(['│', r'\s', '└', '├', '──', ' ', ' ', '─'])
        for i, c in enumerate(string):
            if c not in dont_match:
                return i//4
        raise AssertionError("No non-bar characters found in line.")

    line_nodes = []
    for i, line in enumerate(lines):
        line_nodes.append(
            GraphFromTextNode(line.split(";")[0], get_depth(line), i, ";".join(line.split(";")[1:]) if ";" in line else None)
        )

    cur_node: GraphFromTextNode = line_nodes[0]
    graph = GraphFromText(cur_node)
    queue = [cur_node]  # q <- [root]
    while len(queue) > 0:  # while len(q) > 0:
        parent = queue.pop(0)  #     cur <- q.dq()
        i = parent.index_in_list
        immediate_children = []
        for j in range(i + 1, len(line_nodes)):
            if line_nodes[j].depth <= parent.depth:
                break
            if line_nodes[j].depth - 1 == parent.depth:
                immediate_children.append(line_nodes[j])
        
        if len(immediate_children) > 0:
            parent.is_directory = True

        for child in immediate_children:
            queue.append(child)
            parent.add_child(child)
            child.parent = parent

    return graph


if __name__ == "__main__":
    main()
