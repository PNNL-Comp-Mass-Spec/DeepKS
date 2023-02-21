import traceback, os, types, typing
from termcolor import colored
import termcolor

def informative_exception(e: Exception, top_message="Error: Something went wrong! Error message(s) below:", print_full_tb=True) -> typing.NoReturn:
    print(colored(f"{top_message}\n", "red"))
    print(colored("="*int(0.75*os.get_terminal_size().columns), "red"))
    print(colored(f"  * Error Type: {e.__class__.__name__}", "magenta"))
    print(colored(f"  * Error Message: {e}", "magenta"))
    assert isinstance(e.__traceback__, types.TracebackType)
    traceback_: types.TracebackType = e.__traceback__
    print(colored(f"  * Error Location: {traceback_.tb_frame.f_code.co_filename}, line {traceback_.tb_lineno}", "magenta"))
    print(colored(f"  * Error Function: {traceback.extract_stack(traceback_.tb_frame, limit=1)[-1][2]}", "magenta"))
    print(colored("="*int(0.75*os.get_terminal_size().columns), "red"))
    print()
    if print_full_tb:
        print(colored("Full Traceback:", "magenta"))
        print()
        custom_tb = "".join(traceback.format_stack()[:-2]) + "".join(traceback.format_tb(traceback_))
        print(colored(custom_tb, "magenta"))

    exit(1)

if __name__ == "__main__":
    def fn_a():
        fn_b()
    def fn_b():
        fn_c()
    def fn_c():
        try:
            raise Exception("This is a test exception!")
        except Exception as e:
            informative_exception(e)
    
    fn_a()