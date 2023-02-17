import os, re, pathlib

where_am_i = pathlib.Path(__file__).parent.resolve()
os.chdir(where_am_i)
os.chdir("../../")

def main():
    to_del = []
    for fi in os.listdir("data/"):
        if bool(re.search(r"raw_data_[1-9]+.*\.csv", fi)):
            to_del.append("data/"+fi)
    for fi in os.listdir("data/raw_data/"):
        if bool(re.search(r"kinase_seq.*\.txt", fi)):
            to_del.append("data/raw_data/"+fi)
    
    del_str = "rm -f"
    for x in to_del:
        del_str += f" '{x}'"

    print("The following is a dry run.")
    os.system(f"echo {del_str}")
    response = input("[Y/N to proceed]: ")
    

    if response.lower() == "y":
        os.system(del_str)
    elif response.lower() == "n":
        print("Exiting.")
    else:
        print("Invalid response. Exiting.")

if __name__ == "__main__":
    main()