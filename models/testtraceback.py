import traceback

def main():
    print("step 1")
    traceback.print_stack()
    print("step 2")
    print("step 3")
    traceback.print_stack()
    print("step 4")

main()