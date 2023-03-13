import html2text, traceback, os, types, typing, sys, re, requests, json, textwrap, pathlib, io, pprint
from ..tools.nice_printer import nice_printer
from termcolor import colored



FAKE_TERM_WIDTH = 90
class Capturing(list):
    """https://stackoverflow.com/a/16571630/16158339"""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._stringio = io.StringIO()
        sys.stderr = self._stringio2 = io.StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        self.extend(self._stringio2.getvalue().splitlines())
        del self._stringio    # free up some memory
        del self._stringio2    # free up some memory
        sys.stdout = self._stdout
        sys.stderr = self._stderr

def informative_exception(
    e: Exception,
    top_message="Error: Something went wrong! Error message(s) below.",
    print_full_tb=True,
    exitcode=1,
    fake_root_dir="DeepKS",
) -> typing.NoReturn:
    top_message += "" if not print_full_tb else " Full traceback above."
    print("\n\n", file=sys.stderr)
    assert e.__traceback__ is not None
    traceback_: types.TracebackType = e.__traceback__
    if print_full_tb:
        print(colored("Full Traceback:", "magenta"), file=sys.stderr)
        formatted_stack = traceback.format_tb(traceback_)
        for i, entry in enumerate(formatted_stack):
            regex_search_for_path = re.search('File "(.*)", line', entry)
            assert regex_search_for_path is not None
            extracted_path = regex_search_for_path.group(1)
            try:
                modified_path = "/".join(extracted_path.split("/")[extracted_path.split("/").index(fake_root_dir) :])
            except ValueError:
                modified_path = "/".join(extracted_path.split("/")[-4:])
            custom_text = re.sub('File "(.*)", line ', modified_path + r":", entry)
            inner_lines = ["\n" + "  " * i + x for x in custom_text.strip().split("\n")]
            if i == len(formatted_stack) - 1:
                inner_lines[-1] = re.sub(r"\s{4}(\s{4})(\w)", r"--->\1\2", inner_lines[-1])
            print(colored("".join(inner_lines) + "\n", "magenta"), file=sys.stderr, end="")

        print("", file=sys.stderr)
        # print(colored("".join(traceback.format_tb(traceback_)), "magenta"), file=sys.stderr)
    extr = traceback.extract_tb(traceback_)
    print("\n", file=sys.stderr)
    print(colored(f"{top_message}\n", "red"), file=sys.stderr)
    print(colored("=" * int(0.75 * os.get_terminal_size().columns if FAKE_TERM_WIDTH <= 0 else FAKE_TERM_WIDTH), "red"), file=sys.stderr)
    print(
        colored(
            textwrap.fill(f"  * Error Type: {e.__class__.__name__} (Description: {get_exception_description(e.__class__.__name__)})", width=int(0.75 * os.get_terminal_size().columns if FAKE_TERM_WIDTH <= 0 else FAKE_TERM_WIDTH), subsequent_indent=" "*8),
            "magenta",
        ),
        file=sys.stderr,
    )
    print(colored(f"  * Error Message: {e}", "magenta"), file=sys.stderr)
    assert isinstance(e.__traceback__, types.TracebackType)

    extracted_path = traceback_.tb_frame.f_code.co_filename
    try:
        modified_path = "/".join(extracted_path.split("/")[extracted_path.split("/").index(fake_root_dir) :])
    except ValueError:
        modified_path = "/".join(extracted_path.split("/")[-4:])
    print(colored(f"  * Error Location: {modified_path}:{extr[-1].lineno}", "magenta"), file=sys.stderr)
    print(
        colored(f"  * Error Function: {extr[-1][2]}", "magenta"),
        file=sys.stderr,
    )
    tb_locals = {}
    tb = traceback_.tb_next
    while tb:
        tb_locals = tb.tb_frame.f_locals
        tb = tb.tb_next

    local_variable_nice_io = io.StringIO()
    nice_printer(tb_locals, file=local_variable_nice_io, initial_indent=6)
    local_variable_nice_io.seek(0)
    print(colored(f"  * Local Variables:\n{local_variable_nice_io.read()}", "magenta"), file=sys.stderr)
    print(colored("=" * int(0.75 * os.get_terminal_size().columns if FAKE_TERM_WIDTH <= 0 else FAKE_TERM_WIDTH), "red"), file=sys.stderr)
    print()

    # with Capturing():
    #     raise e
    sys.exit(1)


def get_exception_description(exception_type: str):
    cached = str(pathlib.Path(__file__).parent.resolve()) + "/cached_docs.json"
    if not os.path.exists(cached) or 'REDOWNLOAD' in os.environ:
        base_url = "https://docs.python.org/3/library/exceptions.html"
        r = requests.get(base_url)
        try:
            if r.status_code != 200:
                raise RuntimeError("Failed to get exception description from Python docs.")
            plaintext_version = html2text.html2text(r.text, bodywidth=float("inf"))  # type: ignore
        except Exception as e:
            informative_exception(
                e,
                top_message="Error: Failed to get exception description from Python docs.",
                print_full_tb=False,
                exitcode=1,
            )
        initial_regex = r"_exception _(.*)Â¶\n\n    \n\n(.*)"
        finds = re.findall(initial_regex, plaintext_version)
        assert len(finds) > 0, "Failed to find any exceptions in the Python docs."
        error_type_to_description = dict(finds)
        for key in error_type_to_description:
            error_type_to_description[key] = re.sub(r"\[(`|)(.*?)(`|)]\((.|\n)*?\)", r"\2", error_type_to_description[key])
            error_type_to_description[key] = re.sub('\u00e2\u0080\u0099', r"'", error_type_to_description[key])
            error_type_to_description[key] = re.sub('\u00e2\u0080\u009c', "\"", error_type_to_description[key])
            error_type_to_description[key] = re.sub('\u00e2\u0080\u009d', "\"", error_type_to_description[key])
            error_type_to_description[key] = re.sub('`', r"", error_type_to_description[key])
        doc_dict = error_type_to_description
        json.dump(doc_dict, open(cached, "w"), indent=3)
    else:
        doc_dict = json.load(open(cached, "r"))

    return doc_dict[exception_type] if exception_type in doc_dict else "No description found."


"""
\n\n_exception _(.+)Â¶\n\n    \n\n.*
"""


if __name__ == "__main__":
    def fn_a():
        fn_b()


    def fn_b():
        fn_c()


    def fn_c():
        fn_d()


    def fn_d():
        raise RuntimeError("This is a test exception!")
    
    try:
        fn_a()
    except Exception as e:
        informative_exception(e)
