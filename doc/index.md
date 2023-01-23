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
</style>
<h1 style='font-size:36pt'>DeepKS Manual</h1>

# Getting Started
The bulk of the DeepKS tool is run through Docker. It will essentially run like it would in a virtual machine. This makes dependency management a breeze. Follow the steps below to get started. One neednot clone the DeepKS Git repository to use the tool.

## Colors in this manual
- `This style is used for code the user should run.`
- <code class = "inline-bash-output">This style is used to represent the potential output of a command.</code>

## General Notes Relating to Devices (Read before running any program)
### Does My Computer Have a CUDA-compatible GPU?
If you're not sure, follow the instructions [here](https://askubuntu.com/a/1273434).
### If Running On Personal Computer with CUDA
If you have a CUDA-compatible GPU, you can run the program on your personal computer and take advantage of the GPU. This is the fastest way to run the program (besides using an HPC cluster). 

Most likely, your computer will be running Windows. If this is the case, there is some additional setup involved. If you want to bypass this setup, you can run the program without CUDA on your personal computer or on a HPC cluster (see below). But if you do want to run the program with CUDA on your personal computer, do the following:
1. Go through the steps in the [auxillary help page](https://ben-drucker.gitlab.io/deepks/doc/cuda_installation.html).

In the case you are running Linux, no extra setup is required and you may skip to § [Terminology](#terminology).

### If Running On Personal Computer without CUDA
1. Download Docker here https://www.docker.com/products/docker-desktop/ and follow the installation instructions for your operating system.
### If Running On HPC Cluster
***Note: These instructions are specific to the PNNL "deception" cluster (it assumes `module` and `apptainer` are preinstalled). It also assumes you have an active account.***

1. Open a terminal SSH into the cluster with `ssh <username>@deception.pnnl.gov`, making sure to replace `<username>` with your actual username.
2. Run `module load apptainer` to load Apptainer.
3. Run `apptainer pull benndrucker/deepks:latest` to pull the Docker image.
4. Run `apptainer shell --nv benndrucker/deepks:latest` to start the Docker container (in Apptainer).


<h2 id="terminology"> Terminology </h2>
Please read this explanation: "[An image is a blueprint for a snapshot of a 'system-in-a-system' (similar to a virtual machine).] An instance of an image is called a container...If you start this image, you have a running container of this image. You can have many running containers of the same image." ~ <a href="https://stackoverflow.com/a/23736802/16158339">Thomas Uhrig and Alex Telon's post</a>

## Pull Docker Image
1. 
   - If running on a personal computer, ensure Docker Desktop (Installed above) is running and a terminal is open.
   - If using WSL on Windows, ensure WSL is running.
   - If using HPC cluster, ensure you are SSH'd into the cluster and have run `module load apptainer`.
2. Run the following command to start the docker session: `docker run -it benndrucker/deepks:latest`
3. A command prompt should appear and look like <code class = "inline-bash-output">root@shahash:/#</code>, where `shahash` is a hexadecimal of the Docker Container. You are now inside the Docker Container at the top-level `/` directory. See the steps below to run various programs *from this prompt*.

## Reuse Docker Container
1. To resuse the created container (so that any saved state is available), run `docker ps -a`. This will show a list of all running and previously-created containers.
2. Copy the hash id of the desired container.
3. Run `docker container start <copied hash> -i` (making sure to replace `<copied hash>` with the hexadecimal hash you actually copied). This will give you the command prompt inside the Docker container.

# Running The Programs
***Note: The following steps are run from <u> inside the Docker container</u>. See the steps above to start the Docker container.***

## Using API
## From Command Line
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

## As a Python Import and VS Code integration
### VS Code Integration
To make adding DeepKS to an external codebase, we recommend using VS Code. To do this, follow these steps:
1. Open VS Code.
2. Install the following extensions by searching for them:
   - Dev Containers
   - Remote - Tunnels
   - Remote Explorer
   - Remote - Tunnels
   - WSL (if using WSL)
3. Go back to your terminal, and make sure you're inside the `benndrucker/deepks:latest` Docker container. Then run `/usr/share/code/bin/code-tunnel tunnel` and follow the instructions.

### Importing
It is recommended to clone any external Git repositories to a directory inside the Docker container. Then, you can use VS Code to edit the files. Before importing the modules, you need to tell python where it lives. To do this, run 

```{bash}
printf "\nexport PYTHONPATH=/parent/directory/of/cloned/git/repo/:$PYTHONPATH\n\n" >> ~/.bashrc && source ~/.bashrc
```
making sure to replace `/parent/directory/of/cloned/git/repo/` with the path to the directory containing the cloned Git repository.


To import DeepKS, you can use the following examples (depending on which module(s) you need):
```python
from DeepKS.api.main import make_predictions
import DeepKS.models.multi_stage_classifier as msc
import DeepKS
```
## API Specification
### Functions of DeepKS.api.main:
```python
make_predictions(kinase_seqs, site_seqs, predictions_output_format, verbose, pre_trained_gc, pre_trained_nn):
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
    
    Returns:
        None, or dictionary, or list, depending on `predictions_output_format`
    """

parse_api():
    """Parse the command line arguments.

    Returns:
        dict[str, Any]: Dictionary mapping the argument name to the argument value.
    """
```

# Reproducing Everything From Scratch
TODO — still working on cleaning things up.

## Preprocessing and Data Collection
## Training
The python training scripts contain command line interfaces. However, to make running easier, one can use the bash scripts in the `models` directory. The bash scripts are simply wrappers around the python scripts. The bash scripts are the recommended way to run the training scripts.
1. Run `bash models/train_multi_stage_classifier.sh` to train the multi-stage classifier.
## Evaluating
## Creating Evaluation Diagrams
## Creating Other Diagrams

