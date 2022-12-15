import argparse, re, os, torch, pathlib
where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(os.path.join(os.path.abspath(os.path.join(where_am_i, os.pardir)), "models"))

def parsing():
    global train_filename, val_filename, test_filename
    parser = argparse.ArgumentParser()

    def device(arg_value):
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
        
    parser.add_argument("--device", type=device, help="Specify device. Choices are {'cpu', 'cuda:<gpu number>'}.", metavar='<device>', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--train", type=str, help="Specify train file name", required=True, metavar='<train_file_name.csv>')
    parser.add_argument("--val", type=str, help="Specify validation file name", required=True, metavar='<val_file_name.csv>')
    parser.add_argument("--test", type=str, help="Specify test file name", required=True, metavar='<test_file_name.csv>')

    try:
        args = vars(parser.parse_args())
    except Exception as e:
        print(e)
        exit(1)
    train_filename = args['train']
    val_filename = args['val']
    test_filename = args['test']

    assert 'formatted' in train_filename, "'formatted' is not in the train filename. Did you select the correct file?"
    assert 'formatted' in val_filename, "'formatted' is not in the test filename. Did you select the correct file?"
    assert 'formatted' in test_filename, "'formatted' is not in the test filename. Did you select the correct file?"    
    assert os.path.exists(train_filename), f"Train file '{train_filename}' does not exist."
    assert os.path.exists(test_filename), f"Val file '{val_filename}' does not exist."
    assert os.path.exists(test_filename), f"Test file '{test_filename}' does not exist."
    return args
