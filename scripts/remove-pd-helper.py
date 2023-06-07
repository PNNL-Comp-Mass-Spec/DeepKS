import re, os


def main():
    dir_list = os.listdir("docs/api_pydoctor_docs/")
    for f in dir_list:
        f = os.path.join("docs/api_pydoctor_docs/", f)
        if os.path.isdir(f) or f.split(".")[-1] != "html":
            continue
        with open(f, "r") as fpr:
            fistr = fpr.read()
        with open(f, "w") as fpw:
            repl = re.sub(r'    <meta name="generator" content="pydoctor [\.0-9]+"> \n +\n +<\/meta>', r"", fistr)
            fpw.write(repl)


main()
