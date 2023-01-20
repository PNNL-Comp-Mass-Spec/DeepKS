# Getting Started
The bulk of the DeepKS tool is run through Docker. This makes dependency management a breeze. Follow the steps below to get started. One neednot clone the DeepKS Git repository to use the tool.

## Download Docker
1. Go to https://www.docker.com/products/docker-desktop/ and follow the installation instructions for your operating system.

## Pull Docker Image
<!--TODO: Credentials-->
1. Ensure Docker Desktop (Installed above) is running.
2. Open a terminal (PowerShell on Windows). If needed, see [macOS Instructions](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj8_KLpx9L8AhW_D1kFHSxoCMUQFnoECA0QAQ&url=https%3A%2F%2Fsupport.apple.com%2Fguide%2Fterminal%2Fopen-or-quit-terminal-apd5265185d-f365-44cb-8b09-71a064a42125%2Fmac&usg=AOvVaw38yunYqFSDSP2S9Bs-zTTX) or [Windows Instructions](https://learn.microsoft.com/en-us/powershell/scripting/windows-powershell/starting-windows-powershell?view=powershell-7.3))
3. Run the following command to start the docker session: `docker run -it benndrucker/deepks:0.0.1`
4. A command prompt should appear and look like `shahash#`, where `shahash` is a hexadecimal of the Docker Container. You are now inside the Docker Container. See the steps below to run various programs *from this prompt*.

# Running The Programs
***Note: The following steps are run from <u> inside the Docker container</u>. See the steps above to start the Docker container.***
## General Notes Relating to Devices (Read before running any program)
### Does My Computer Have a CUDA-compatible GPU?
If you're not sure, follow the instructions [here](https://askubuntu.com/a/1273434).
### Running On Personal Computer with CUDA
If you have a CUDA-compatible GPU, you can run the program on your personal computer and take advantage of the GPU. This is the fastest way to run the program (besides using an HPC cluster). To do so, you must have the CUDA Toolkit installed. You can download it [here](https://developer.nvidia.com/cuda-downloads). You must also have the NVIDIA drivers installed. You can download them [here](https://www.nvidia.com/Download/index.aspx?lang=en-us). <!-- TODO Verify -->
### Running On Personal Computer without CUDA
No extra setup steps are necessary, but evaluations in the deep learning model will be much slower.
### Running On HPC Cluster
TODO -- if there is interest

## Using API
### From Command Line
The Command Line Interface is the main way to query the deep learning model. The API is a submodule of `DeepKS`. (Henceforth referred to as `DeepKS.api`.) The `DeepKS.api` module, itself contains a submodule `main`. (Henceforth referred to as `DeepKS.api.main`). This is the main "entrypoint" for running queries. Because of various Python specifications, `DeepKS.api.main` must be run as a module from _outside_ the `/DeepKS` directory. Hence, to run from the command line in the Docker container, run 

```bash
cd /
python -m DeepKS.api.main [options]
```
where `[options]` are the options you wish to pass to the program. To see the required and optional arguments, run `python -m DeepKS.api.main --help`.

When you run this, it will print 

```bash
usage: python -m DeepKS.api.main [-h] (-k <kinase sequences> | -kf <kinase sequences file>)
                                 (-s <site sequences> | -sf <site sequences file>)
                                 [-p {in_order,dictionary,in_order_json,dictionary_json}] [-v]
                                 [--pre_trained_nn <pre-trained neural network file>]
                                 [--pre_trained_gc <pre-trained group classifier file>]
```
- Anything in square brackets is optional.
- For each instance of round parentheses, you must provide one of the options between "`|`". 
- Curly braces show available options for a flag.

With that in mind, here are some examples of how to run the program (make sure to be in the top-level `/` directory):

```bash
python -m DeepKS.api.main -kf my/kinase/sequences.txt -sf my/site/sequences.txt -p in_order_json -v True

python -m DeepKS.api.main -k KINASESEQ1,KINASESEQ2,KINASESEQ3 -s SITESEQ1,SITESEQ2,SITESEQ3 -p dictionary

python -m DeepKS.api.main -kf my/kinase/sequences.txt -s SITESEQ1,SITESEQ2,SITESEQ3 -p in_order -v False

python -m DeepKS.api.main -kf my/kinase/sequences.txt -sf my/site/sequences.txt
```

### As a Python Import
It is recommended to clone any external Git repositories to a directory inside the Docker container...TODO -- incomplete

#### API Specification
##### Functions of DeepKS.api.main:
```python
    make_predictions(
        kinase_seqs: list[str],
        site_seqs: list[str],
        predictions_output_format: str = "in_order",
        verbose: bool = True,
        pre_trained_gc: str = "data/bin/deepks_gc_weights.0.0.1.pt",
        pre_trained_nn: str = "data/bin/deepks_nn_weights.0.0.1.pt",
    ):
        """Make a target/decoy prediction for a kinase-substrate pair.

        Args:
            kinase_seqs (list[str]): The kinase sequences. Each must be <= 4128 residues long.
            site_seqs ([str]): The site sequences. Each must be 15 residues long.
            predictions_output_format (str, optional): The format of the output. Defaults to "in_order".
                - "in_order" returns a list of predictions in the same order as the input kinases and sites.
                - "dictionary" returns a dictionary of predictions, where the keys are the input kinases and sites and the values are the predictions.
                - "in_order_json" outputs a JSON string (filename = ../out/current-date-and-time.json of a list of predictions in the same order as the input kinases and sites.
                - "dictionary_json" outputs a JSON string (filename = ../out/current-date-and-time.json) of a dictionary of predictions, where the keys are the input kinases and sites and the values are the predictions.
            verbose (bool, optional): Whether to print predictions. Defaults to True.
            pre_trained_gc (str, optional): Path to previously trained group classifier model state. Defaults to "data/bin/deepks_weights.0.0.1.pt".
            pre_trained_nn (str, optional): Path to previously trained neural network model state. Defaults to "data/bin/deepks_weights.0.0.1.pt".
        """
```

## Reproducing Everything From Scratch
TODO -- still working on cleaning things up.
### Preprocessing and Data Collection
### Training
### Evaluating
### Creating Evaluation Diagrams

