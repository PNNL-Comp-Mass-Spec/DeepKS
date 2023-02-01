<style>
    pre.bash-output.bash-output{
        background-color: #ebe9c27f;
    }
    code.inline-bash-output{
        background-color: #ebe9c27f;
    }
    code{
        background-color: rgba(220, 220, 220, 0.4);
        padding: 1px 3px;
        border-radius: 5px;
    }
    pre code{
        background-color: transparent;
        padding: 0;
        border-radius: 0;
    }
    h1{
        border-bottom-width: 2px;
    }
    
    h2{
        border-bottom-width: 1px;
        border-bottom-color: #00000040;
        border-bottom-style: solid;
    }

    h3{
        border-bottom-width: 1px;
        border-bottom-color: #00000040;
        border-bottom-style: dashed;
    }

    ol {
        margin: 0;
        padding: 0;
        list-style-type: none;
    }

    ol > li:before {
        counter-increment: step-counter;
        content: counter(step-counter) ". ";
        margin-right: 3px;
        font-size: 18px;
        font-weight: bold;
        /* padding: 3px 3px; */
    }

    ol:first-of-type {
        counter-reset: step-counter;
    }

</style>

# Quickstart (Gets things up and running, but does not explain the theory behind the tool)
This guide does not go into all the possible options and methods of DeepKS. For a more detailed explanation of the theory behind DeepKS, please refer back to the [main manual](https://ben-drucker.gitlab.io/deepks/).
## Instructions
1. [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Run Docker Desktop
3. Open up Terminal (Mac, Linux, Windows — preferred) or Powershell (Windows) or Command Prompt (Windows)
4. The interface — in an attempt to update the git repository, will ask for your username and password. Fill that in.
5. Run the following command:
    ```bash
    docker run -it --name deepks-container --network host --hostname deepks-container benndrucker/deepks
    ```
6. You should see this prompt: <code class = "inline-bash-output">(base) //deepks-container// [/] ▷ </code>. You are now in the DeepKS container. You must run DeepKS commands from here.
7. Run the following command:
```bash
python3 -m DeepKS.examples
```
This will run a few examples of DeepKS. You should see the following output (with more lines at the end):
<pre class = "bash-output bash-output">
Info: This is an example script for DeepKS. To inspect the sample input files, check the 'examples/sample_inputs' directory.

Info: [Example 1/3] Simulating the following command line from `DeepKS/`:

<span style = "color: blue">python3 -m DeepKS.api.main -kf ../tests/sample_inputs/kins.txt -sf ../tests/sample_inputs/sites.txt -p dictionary -v</span>

Status: Loading Modules...
Status: Loading previously trained models...
Status: Beginning Prediction Process...
Prediction Step [1/2]: Sending input kinases to group classifier
Prediction Step [2/2]: Sending input kinases to individual group classifiers, based on step [1/2]
Group Neural Network Evaluation Progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  9.61it/s]

<<<<<<<<<<<<<<<< REQUESTED RESULTS >>>>>>>>>>>>>>>>

[{'kinase': 'NEGWVSQFKQYLVDTDHNVTRRTAAYLFHYVARTWPECVHIDNMDMTNIVDFHCRVIQLSKNFDFTWCFNCWWWWRGGMEAKPEYYYPHLMIDEMQRCID',
'site': 'AYGEKEHCANCHNIV',
'prediction': 'Decoy'},
{'kinase': 'RIDNKWVLPRKSWQQNEEAMCAPVNCAHFKTYPINHYQYWGAAACCTFTGHISKGEWETPYMKSFFMTEPMYQSKTSGEQKRSTEAWGGHLWFHPTWHHD',
'site': 'CHEFIPAKIPTNNIS',
'prediction': 'Decoy'},

...
</pre>
7. If this worked, you are successful!
8. To close the docker container at any time, run `exit`. (Before doing so, make note of or copy the hash in the prompt.) This will close the container and return you to your normal terminal.
9. To run DeepKS again, simply run the command `docker start -t <hash>` where `<hash>` is the hash of the container you just closed. The hash is the one you took a note of or copied in step 8. If you lost the hash, run `docker ps -a` to see a list of all containers (and hashes thereof) you have run.