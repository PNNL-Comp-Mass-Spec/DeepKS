"""Reimplementation of the custom tree maker, which produces an HTML-based tree with directory and file descriptions."""

import json, os, re
from ..config.join_first import join_first

name_to_node: dict[str, tuple[str, list[str], list[str]]] = {}


def get_walk(
    root=join_first("", 1, __file__), excluded_file_name=[], included_file_name=[], excluded_dir=[], included_dir=[]
):
    the_walk = list(os.walk(root))
    for node in the_walk:
        name_to_node[node[0]] = node
    return walk_to_dict(the_walk[0], excluded_file_name, included_file_name, excluded_dir, included_dir)


def walk_to_dict_helper(
    node: tuple[str, list[str], list[str]], excluded_file_name, included_file_name, excluded_dir, included_dir
):
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
        wtdh = walk_to_dict_helper(
            name_to_node[child_dir], excluded_file_name, included_file_name, excluded_dir, included_dir
        )
        for k, v in wtdh.items():
            path = child_dir.split("/")
            ii = list(reversed(path)).index("..")
            path = list(reversed(list(reversed(path))[:ii] + ["DeepKS"]))
            desc = path_tup_to_desc.get(tuple(path), "")

            path = list(reversed(list(reversed(path))[:ii] + ["DeepKS"]))
            desc = path_tup_to_desc.get(tuple(path), "")
            if os.path.exists(os.path.join(node[0], k, "__init__.py")):
                # print(os.path.join(node[0], k, "__init__.py"))
                module_from_path = ".".join(path[:-1] + [path[-1].split(".")[0]])
                desc = (
                    f"See <a target=\"_top\", href=\"{f'api_pydoctor_docs/{module_from_path}.html'}\"><code"
                    f" class='only-border'>{module_from_path}</code></a>."
                )
            desc = re.sub(r"¡(.*)¡", r"<span class='warn'>\1</span>", desc)
            partial_dict[node[0]][k] = {"node": v, "desc": desc, "path": path}
            path_to_desc[tuple(path)] = path_tup_to_desc.get(tuple(path), "")

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
        path = os.path.join(node[0], child_file).split("/")
        ii = list(reversed(path)).index("..")
        path = list(reversed(list(reversed(path))[:ii] + ["DeepKS"]))
        desc = path_tup_to_desc.get(tuple(path), "")
        if re.search(r"\.py$", path[-1]):
            # print
            module_from_path = ".".join(path[:-1] + [path[-1].split(".")[0]])
            desc = (
                f"See <a target=\"_top\" href=\"{f'api_pydoctor_docs/{module_from_path}.html'}\"><code"
                f" class='only-border'>{module_from_path}</code></a>."
            )
        desc = re.sub(r"¡(.*)¡", r"<span class='warn'>\1</span>", desc)
        partial_dict[node[0]][f"{child_file.split('/')[-1]}"] = {
            "desc": desc,
            "path": path,
        }
        path_to_desc[tuple(path)] = path_tup_to_desc.get(tuple(path), "")

    return dict({k.split("/")[-1]: v for k, v in partial_dict.items()})


def format_into_html(existing_str: str) -> str:
    return f"""<!DOCTYPE html>
    <html>

    <head>
        <meta charset="UTF-8">
        <title></title>
        <style>
            /* From extension vscode.github */
            /*---------------------------------------------------------------------------------------------
    *  Copyright (c) Microsoft Corporation. All rights reserved.
    *  Licensed under the MIT License. See License.txt in the project root for license information.
    *--------------------------------------------------------------------------------------------*/

            .vscode-dark img[src$=\\#gh-light-mode-only],
            .vscode-light img[src$=\\#gh-dark-mode-only] {{
                display: none;
            }}
        </style>

        <link rel="stylesheet"
            href="https://cdn.jsdelivr.net/gh/Microsoft/vscode/extensions/markdown-language-features/media/markdown.css">
        <link rel="stylesheet"
            href="https://cdn.jsdelivr.net/gh/Microsoft/vscode/extensions/markdown-language-features/media/highlight.css">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe WPC', 'Segoe UI', system-ui, 'Ubuntu', 'Droid Sans', sans-serif;
                font-size: 14px;
            }}
        </style>
        <style>
            .task-list-item {{
                list-style-type: none;
            }}

            .task-list-item-checkbox {{
                margin-left: -20px;
                vertical-align: middle;
                pointer-events: none;
            }}

            .inner-div {{
                display: inline-block;
            }}
        </style>

    </head>

    <body class="vscode-body vscode-light" style="white-space: nowrap;">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fira+Code&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@600&display=swap');

            div {{
                margin: 10px;
                font-size: 12px;
            }}

            div > div > code {{
                vertical-align: baseline;
            }}

            a > code.only-border {{
                vertical-align: baseline;
            }}

            code {{
                font-family: 'Fira Code';
                color: rgb(40, 40, 40);
                font-style: normal;
                border-radius: 2.5px;
                display: inline-block;
                padding-block: 1px;
                padding-inline: 1px;
                white-space: pre;
                line-height: 1em;
                vertical-align: middle;
            }}
            
            .lines {{
                padding-block: 0px;
                padding-inline: 0px;
                font-size: 19px;
                font-weight: 300;
            }}
            .no-col {{
                background-color: transparent;
                color: inherit;
            }}

            .only-border {{
                background-color: transparent;
                color: rgba( 0, 0, 128 , 1);
                text-decoration: underline;
                text-decoration-thickness: 1px;
            }}

            div {{
                font-style: oblique;
                padding: 0px;
                margin: 0px;
            }}
            html, body{{
                margin: 0px;
                padding: 0px;
                color: black;
                display: inline-block;
            }}

            div.even, div.oddd {{
                padding-right: 10px;
            }}

            p {{
                margin: 0px;
                font-style: oblique;
                display: inline;
                vertical-align: middle;
            }}

            .even {{
                background-color: #e1e4e8;
            }}
            .oddd {{
                background-color: #f5f5f5;
            }}
            .dir {{
                font-weight: 600;
            }}
            .warn {{
                color: rgba(255, 60, 0, 0.871);
                font-weight: bold;
                text-decoration: underline;
                text-decoration-style: wavy;
                text-decoration-thickness: 1px;
            }}

        </style>
        {existing_str}

    </body>

    </html>
    """


def walk_to_dict(
    root_node: tuple[str, list[str], list[str]], excluded_file_name, included_file_name, excluded_dir, included_dir
):
    return walk_to_dict_helper(root_node, excluded_file_name, included_file_name, excluded_dir, included_dir)


def traverse_nested(nested: dict):
    path = ["DeepKS"]
    nested = {k: {"node": v, "desc": path_tup_to_desc.get(tuple(path), ""), "path": path} for k, v in nested.items()}
    path_to_desc[tuple(path)] = path_tup_to_desc.get(tuple(path), "")
    return traverse_nested_helper(nested, 0, set())


bg_class = "even"


def traverse_nested_helper(nested: dict, level: int, no_vert_levels: set[int]) -> str:
    global bg_class
    VBAR = "│"
    VBARHBAR = "├ "
    ENDBAR = "└ "
    # DIRSYMB = "⦿ "
    DIRSYMB = ""
    DESCSYMB = " ❖ "
    build_up = ""

    def sort_key(x: tuple) -> tuple:
        if "node" not in x[1]:
            return ("", x[0])
        else:
            return (x[0], x[0])

    for i, (k, v) in enumerate(ll := sorted(list(nested.items()), key=sort_key)):
        if bg_class == "even":
            bg_class = "oddd"
        else:
            bg_class = "even"
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

        DESCSYMB_ = DESCSYMB
        DIRSYMB_ = DIRSYMB
        if not v["desc"]:
            DESCSYMB_ = ""
        is_dir = "node" in v
        add_on_set = set()
        if is_dir:
            add_on_set: set[int]
            if i + 1 == len(ll):
                add_on_set = set.union(no_vert_levels, {level})
            else:
                add_on_set = no_vert_levels
            bolded = f"<span style='font-weight: bold; font-size: 14px;'>{k}</span>"
        else:
            DIRSYMB_ = ""
            bolded = k

        build_up += (
            f"<div class='{bg_class}'><code class='no-col lines'>"
            + spaces
            + syb
            + "</code>"
            + "<code class='no-col'>"
            + DIRSYMB_
            + bolded
            + DESCSYMB_
            + "</code>"
            + "<div style='display: inline; vertical-align: middle;'>"
            + v.get("desc", "")
            + "</div>"
            + "</div>"
            + "\n"
        )
        if is_dir:
            build_up += traverse_nested_helper(v["node"], level + 1, add_on_set)
    return build_up


def main():
    global path_to_desc, path_tup_to_desc
    with open(join_first("docs/path_to_desc.json", 1, __file__)) as f:
        legacy = json.load(f)
    path_to_desc = {}
    path_tup_to_desc = {}
    for k, v in legacy.items():
        if v is not None:
            path_tup_to_desc[eval(k)] = v.strip()
    excluded_both = [r"__pycache__", r"__init__", r"vscode", r"\.git", r"docs/api_pydoctor_docs/[^\.]+", r"out/[^\.]+"]
    excluded_file_name = [r"\.pyc", r"\.DS_Store", r"\.gitig-*", r"tree.txt"] + excluded_both
    included_file_name = [
        r"docs/api_pydoctor_docs/index.html",
        r"results_2023-05-03@19`00`32.0@-04`00.json",
        r"results_2023-05-03@19`00`34.4@-04`00.csv",
        r"results_2023-05-03@19`00`36.6@-04`00.sqlite",
    ]
    excluded_dir = excluded_both
    included_dir = []

    tree_trav = get_walk(
        join_first("", 1, __file__), excluded_file_name, included_file_name, excluded_dir, included_dir
    )
    # with open("tree_data.json", "w") as f:
    #     json.dump(tree_trav, f, indent=3)
    pretty = traverse_nested(tree_trav)
    with open(join_first("docs/tree.html", 1, __file__), "w") as f:
        f.write(format_into_html(pretty))
    with open(join_first("docs/path_to_desc.json", 1, __file__), "w") as f:
        json.dump({str(k): v for k, v in path_to_desc.items()}, f, indent=3)


if __name__ == "__main__":  # pragma: no cover
    get_walk()
