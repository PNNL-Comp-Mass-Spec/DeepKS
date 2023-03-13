import argparse, re, os, torch, pathlib, warnings
from typing import Union
where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(os.path.join(os.path.abspath(os.path.join(where_am_i, os.pardir)), "models"))

def parsing() -> dict[str, Union[str, None]]:
    print("Progress: Parsing Arguments")
    global train_filename, val_filename, test_filename
    parser = argparse.ArgumentParser()


    def device_eligibility(arg_value):
        try:
            assert(bool(re.search("^cuda(:|)[0-9]*$", arg_value)) or bool(re.search("^cpu$", arg_value)))
            if "cuda" in arg_value:
                if arg_value == "cuda":
                    return arg_value
                cuda_num = int(re.findall("([0-9]+)", arg_value)[0])
                assert(0 <= cuda_num <= torch.cuda.device_count())
        except Exception:
            raise argparse.ArgumentTypeError(f"Device '{arg_value}' does not exist. Choices are {'cpu', 'cuda[:<gpu #>]'}.")
        
        return arg_value
    # determine_device_eligibility_wrapper = lambda x: device(x)
        
    parser.add_argument("--device", type=str, help="Specify device. Choices are {'cpu', 'cuda:<gpu number>'}.", metavar='<device>', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--train", type=str, help="Specify train file name", required=False, metavar='<train_file_name.csv>')
    parser.add_argument("--val", type=str, help="Specify validation file name", required=False, metavar='<val_file_name.csv>')
    parser.add_argument("--test", type=str, help="Specify test file name", required=False, metavar='<test_file_name.csv>')
    parser.add_argument("--load", type=str, help="Specify path from which to load", required=False, metavar='<load/file/name>')
    parser.add_argument("--load-include-eval", type=str, help="Specify path from which to load", required=False, metavar='<load/file/name>')
    parser.add_argument('-s', action='store_true', help="Include to save state", required=False)
    parser.add_argument('-c', action='store_true', help="Include to create model binaries", required=False)

    try:
        args = vars(parser.parse_args())
    except Exception as e:
        print(e)
        exit(1)

    device_eligibility(args['device'])

    if args['load_include_eval'] is None:
        test_filename = args['test']
        if args['load'] is not None:
            load_filename = args['load']
            assert os.path.exists(load_filename), f"Load file '{load_filename}' does not exist."
            assert all([args[x] is None for x in ['train', 'val']]), "If specifying --load argument, cannot specify --train and --val arguments."
            # for f in [test_filename]: # FIXME
            #     assert 'formatted' in f, "'formatted' is not in the train filename. Did you select the correct file?"
            #     assert os.path.exists(f), f"Test file '{f}' does not exist."
        else:
            train_filename = args['train']
            val_filename = args['val']
            assert all([args[x] is not None for x in ['train', 'val']]), "If not specifying --load argument, must` specify --train and --val arguments."
            for f in [train_filename, val_filename]:
                try:
                    assert 'formatted' in f, "'formatted' is not in the train filename. Did you select the correct file?"
                except AssertionError as e:
                    warnings.warn(str(e), UserWarning)
                assert os.path.exists(os.path.expanduser(f)), f"Input file '{f}' does not exist."
    else:
        load_filename = args['load_include_eval']
        assert os.path.exists(load_filename), f"Load file '{load_filename}' does not exist."
        assert all([args[x] is None for x in ['train', 'val', 'test', 'load']]), "If specifying --load-include_eval argument, cannot specify --train, --val, --test, nor --load arguments."
    return args
