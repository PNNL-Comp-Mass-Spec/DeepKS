from __future__ import annotations
import collections

import os, json, re, pprint
from sys import argv
from typing import Union

def main():
    if (os.path.exists("tree.txt") and len(argv) == 1) or (len(argv) > 1 and argv[1] != "-f"):
        print("Use -f to forcefully overwrite tree.txt")
        exit(1)

    exitcode = os.system("tree -o tree.txt -I \"*.pyc\" -I \"__pycache__\" -I \"__init__.py\"")
    if exitcode:
        print(f"`tree` didn't run correctly (exit code {exitcode})")
        exit(1)

    with open("tree_description.txt", "r") as d:
        description = d.readlines()

    with open("tree.txt", "r") as t:
        lines = t.readlines()

    new_lines = []
    tree_repr = graph_from_text(lines).get_pre_order_trav()
    tree_description_path_dict = graph_from_text(description).get_dict_repr()
    for orig_line, node in zip(lines, tree_repr):
        node_inner = node
        path_to_node = [node_inner.text]
        while node_inner.parent is not None:
            path_to_node.append(node_inner.parent.text)
            node_inner = node_inner.parent
        path_to_node = tuple(path_to_node)
        desc = ""
        if path_to_node in tree_description_path_dict:
            desc = tree_description_path_dict[path_to_node].description
        if desc != "":
            new_lines.append(orig_line.replace("\n", "") + " ⧐ " + desc.replace("\n", ""))
    
    with open("tree.txt", "w") as t:
        t.write("\n".join(new_lines))



class GraphFromTextNode:
    def __init__(self, text: str, depth: int, index_in_list: int, description=""):
        self.children: list[GraphFromTextNode] = []
        self.parent: Union[GraphFromText, None] = None
        self.text = text
        self.text = re.sub(r"[\n│\s└├──]", "", self.text)
        self.description=description
        self.depth = depth
        self.index_in_list = index_in_list
    def __str__(self):
        return "    "*self.depth + f"<< {self.text} >>"
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
        self._repr_str_build_up.append(node.__str__())
        if len(node.children) > 0:
            for child in node.children:
                self._str_helper(child)
            
    def __repr__(self):
        return self.__str__()
    
    def get_pre_order_trav(self):
        self.res = []
        self._get_pre_order_trav_helper(self.root)
        ret = self.res
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
        
        self.res_d[tuple(abs_path_to_node)] = node

        if len(node.children) > 0:
            for child in node.children:
                self._get_dict_repr_helper(child)
        else:
            return node


    # def get_path_to_node(self, target):
    #     return self._get_path_to_node_helper(self.root, target)
    
    # def _get_path_to_node_helper(self, node, target):
    #     for child in node.children:
    #         if child.text == target.text:



def graph_from_text(lines) -> GraphFromText:
    get_depth = lambda x: 0 if re.match(r"^\.;*.*", x) else len(re.findall(r"│", x)) + 1
    line_nodes = []
    for i, line in enumerate(lines):
        line_nodes.append(GraphFromTextNode(line.split(";")[0], get_depth(line), i, line.split(";")[-1] if ";" in line else ""))
    
    cur_node: GraphFromTextNode = line_nodes[0]
    graph = GraphFromText(cur_node)
    queue = [cur_node] # q <- [root]
    while len(queue) > 0: # while len(q) > 0:
        parent = queue.pop(0) #     cur <- q.dq()
        i = parent.index_in_list
        immediate_children = []
        for j in range(i + 1, len(line_nodes) - 1):
            if line_nodes[j].depth <= parent.depth:
                break
            if line_nodes[j].depth - 1 == parent.depth:
                immediate_children.append(line_nodes[j])

        for child in immediate_children:
            queue.append(child)
            parent.add_child(child)
            child.parent = parent

    return graph

if __name__ == "__main__":
    main()
