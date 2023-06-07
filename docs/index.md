<style>
    :root{
        --pctg: 25%;
    }
    iframe {
        border-width: 2px;
        border-color: black;
        border-style: solid;
        border-radius: 6px;
        padding: 10px;
        box-sizing: border-box 
    }
    .fancy-link {
        font-size: 52px;
        color: black;
        text-align: center;
        vertical-align: center;
        background-image: url("API Docs Icon.png");
        width:200px; 
        height:200px;
        border-width:2px;
        border-color:black; 
        border-radius:5px; 
        border-style:solid; 
        transition: cubic-bezier(.18,.89,.32,1.28) 0.125s;
        background-size: contain;
        display: flex;
        justify-content: center;
        align-content: center;
        flex-direction: column;
    }
    .fancy-link:hover {
        text-decoration: unset;
        transform: scale(1.1);
        opacity: 0.68;
    }
    .no-underline {
        text-decoration: none;
    }
    .no-underline:hover {
        text-decoration: none;
    }
    pre.bash-output.bash-output{
        background-color: #ebe9c27f;
    }
    code.inline-bash-output{
        background-color: #ebe9c27f;
    }
    code{
        background-color: rgba(220, 220, 220, 0.5);
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
        margin-top: 50px;
        margin-bottom:10px;
        margin-left:0px;
        margin-right:0px;
    }
    h1:first-of-type{
        border-bottom-width: 2px;
        margin-top: 10px;
        margin-bottom:10px;
        margin-left:0px;
        margin-right:0px;
    }
    
    h2{
        border-bottom-width: 1px;
        border-bottom-color: #00000040;
        border-bottom-style: solid;
        margin-top: 25px;
    }

    h3{
        border-bottom-width: 1px;
        border-bottom-color: #00000040;
        border-bottom-style: dashed;
        margin-top: 25px;
    }

    h4, h5, h6{
        margin-top: 25px;
    }

    ul {
        padding-left: 12px;
    }

    /* .tab-cell {
        vertical-align: top;
    } */

    .tab-cell-inner {
        border-radius: 3px;
        border-style: solid;
        border-width: 1.5px;
        padding: 12px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    html,
    body {
        padding: unset;
        margin: unset;
        border: unset;
        background: rgba(255, 255, 255, 1);
    }
</style>
<div class="tab-cont">
<div class="tab-cell" style="float:left; width:var(--pctg);">
<div class="tab-cell-inner" style="margin-left:12px; margin-right:6px;", id='outer'>

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Quickstart (Gets things up and running, but does not explain the tool — the rest of the manual goes in depth)](#quickstart-gets-things-up-and-running-but-does-not-explain-the-tool--the-rest-of-the-manual-goes-in-depth)
  - [Colors in this manual](#colors-in-this-manual)
  - [Abbreviations in this project](#abbreviations-in-this-project)
- [General Notes Relating to Devices (Read before running any program)](#general-notes-relating-to-devices-read-before-running-any-program)
  - [Does My Computer Have a CUDA-compatible GPU?](#does-my-computer-have-a-cuda-compatible-gpu)
  - [Follow **one** of the cases below.](#follow-one-of-the-cases-below)
    - [Case A: Running On Personal Computer with CUDA](#case-a-running-on-personal-computer-with-cuda)
    - [Case B: Running On Personal Computer without CUDA](#case-b-running-on-personal-computer-without-cuda)
    - [Case C: Running On HPC Cluster](#case-c-running-on-hpc-cluster)
- [Getting Started with Docker](#getting-started-with-docker)
  - [Follow **one** of the cases below.](#follow-one-of-the-cases-below-1)
    - [Case A/B: Running On Personal Computer](#case-ab-running-on-personal-computer)
    - [Case C: Running on HPC cluster](#case-c-running-on-hpc-cluster-1)
- [Trying Examples](#trying-examples)
- [Full `DeepKS` Specification](#full-deepks-specification)
- [Using Command Line API](#using-command-line-api)
    - [Adding to new code and importing](#adding-to-new-code-and-importing)
      - [VS Code Integration](#vs-code-integration)
      - [Importing](#importing)
  - [Running tests](#running-tests)
- [File Explainer](#file-explainer)
- [Troubleshooting](#troubleshooting)
- [Reproducing Everything From Scratch](#reproducing-everything-from-scratch)
  - [Preprocessing and Data Collection](#preprocessing-and-data-collection)
  - [Training](#training)
  - [Evaluating](#evaluating)
  - [Creating Evaluation Diagrams](#creating-evaluation-diagrams)
  - [Creating Other Diagrams](#creating-other-diagrams)
</div>
</div>
<div class="tab-cell" style="float:right; width:calc(100% - var(--pctg));">
<div class="tab-cell-inner" style="margin-right:12px; margin-left:6px;">

<h1 style='font-size:36pt'>DeepKS Manual</h1>

<span style='font-size:15pt'> The bulk of the DeepKS tool is run through Docker. It will essentially run like it would in a virtual machine. This makes dependency management a breeze and ensures the program will run exactly the same way on every computer. Follow the steps below to get started. One need not clone the DeepKS Git repository to use the tool. </span>

# Quickstart (Gets things up and running, but does not explain the tool — the rest of the manual goes in depth)
1. See the [Quickstart guide](quickstart.html).

## Colors in this manual
- `This style is used for code the user should run.`
- <code class = "inline-bash-output">This style is used to represent the desired output of a command.</code>

## Abbreviations in this project
- NN = Neural Network
- GC = Group Classifier
- KS = Kinase Substrate
- HPC = High Performance Computing
- CUDA = Compute Unified Device Architecture (NVIDIA GPU platform)

# General Notes Relating to Devices (Read before running any program)
## Does My Computer Have a CUDA-compatible GPU?
If you're not sure, follow the instructions [here](https://askubuntu.com/a/1273434).
## Follow **one** of the cases below.
### Case A: Running On Personal Computer with CUDA
If you have a CUDA-compatible GPU, you can run the program on your personal computer and take advantage of the GPU. This is the fastest way to run the program (besides using an HPC cluster). 

Most likely, your computer will be running Windows. If this is the case, there is some additional setup involved. If you want to bypass this setup, you can run the program without CUDA on your personal computer or on a HPC cluster (see below). But if you do want to run the program with CUDA on your personal computer, do the following:
1. Go through the steps in the [auxillary help page](cuda_installation.html).

### Case B: Running On Personal Computer without CUDA
1. Download Docker here https://www.docker.com/products/docker-desktop/ and follow the installation instructions for your operating system.
### Case C: Running On HPC Cluster
***Note: These instructions are specific to the PNNL "deception" cluster (it assumes `module` and `apptainer` are preinstalled and configured appropriately). It also assumes you have an active account.***

0. Make sure you are connected to the cluster's VPN. (In the case of PNNL, make sure you are on campus or connected to the Onekey VPN.)
1. Open a terminal SSH into the cluster with `ssh <username>@deception`, making sure to replace `<username>` with your actual username.
2. Download the interactive Slurm script by running `cd ~ && wget https://raw.githubusercontent.com/Ben-Drucker/DeepKS/main/scripts/hpc/.interactive_slurm_script.py`
3. Run `python .interactive_slurm_script.py`. This will request an interactive session on a compute node.
4. Ensure your session is loaded (i.e., that you are now in a terminal on the HPC. You can check this by running `hostname`. It should no longer be `deception0X`.)
5. Run `module load apptainer` to load Apptainer.




# Getting Started with Docker
<h2 id="terminology"> Terminology </h2>
Please read this explanation: "[An image is a blueprint for a snapshot of a 'system-in-a-system' (similar to a virtual machine).] An instance of an image is called a container...If you start this image, you have a running container of this image. You can have many running containers of the same image." ~ <a href="https://stackoverflow.com/a/23736802/16158339">Thomas Uhrig and Alex Telon's post</a>

## Follow **one** of the cases below.
### Case A/B: Running On Personal Computer
1. If using WSL, make sure it is running. Otherwise, ensure Docker Desktop (Installed above) is running and a terminal is open.
2. Run the following command to start the docker session: `docker run -it --name deepks-container --network host --hostname deepks-container benndrucker/deepks`.
   1. The name `deepks-container` is arbitrary. You can name it whatever you want. In fact, if you need to run multiple instances of the Docker container, you must name them differently.
3. The interface — in an attempt to update the git repository, will ask for your username and password. Fill that in.
4. A command prompt should appear and look like <code class = "inline-bash-output">(base) //root@deepks-container// [/] ▷ </code>. You are now inside the Docker Container at the top-level `/` directory. See the steps below to run various programs *from this prompt*.
5. To reuse the created container (so that any saved state is available), run `docker ps -a`. This will show a list of all running and previously-created containers.
6. Note the name of the container you want to start.
7. Run `docker container start <noted name> -i` (making sure to replace `<noted name>` with the container name you noted). This will give you the command prompt inside the Docker container.

### Case C: Running on HPC cluster
Because we will use Apptainer to run the docker container, the commands are different from cases A/B.
1. Ensure Apptainer is loaded (`module load apptainer`).
2. Run `apptainer scripts --sandbox deepks-latest.sif docker://benndrucker/deepks:latest` to scripts the Apptainer-compatible `.sif` directory. This will take a while (~30-60+ mins) depending on your internet connection and processor speed. You may get `xattr`-related warnings, but these are fine.
3. Copy necessary Nvidia files using the following script (ensuring you are in the same directory as `deepks-latest.sif`):
```{bash}
cp /usr/bin/nvidia-smi deepks-latest.sif/usr/bin/
cp /usr/bin/nvidia-debugdump deepks-latest.sif/usr/bin/
cp /usr/bin/nvidia-persistence deepks-latest.sif/usr/bin/
cp /usr/bin/nvidia-cuda-mps-server deepks-latest.sif/usr/bin/
cp /usr/bin/nvidia-cuda-mps-control deepks-latest.sif/usr/bin/
```
4. The top-level directory structure of `deepks-latest.sif` must mirror the native root directory. Thus, when running the next command, you may receive binding or mounting errors. The solution is creating "fake," empty directories at the top level of `deepks-latest.sif`. [PNNL SPECIFIC] You need to make the "fake" directory `/people` in `deepks-latest.sif` by running `mkdir deepks-latest.sif/people`.
5. Run `apptainer shell --nv --writable --fakeroot deepks-latest.sif` to start the Docker container (in Apptainer). This may give three warnings about Nvidia and mounts:
<pre class = "bash-output bash-output">
<span style="color:#AAA956">WARNING:</span> nv files may not be bound with --writable
<span style="color:#AAA956">WARNING:</span> Skipping mount /etc/localtime [binds]: /etc/localtime doesn't exist in container
<span style="color:#AAA956">WARNING:</span> Skipping mount /var/run/nvidia-persistenced/socket [files]: /var/run/nvidia-persistenced/socket doesn't exist in container
</pre>
These don't seem to cause any issues.

6. Change directory to `/` (i.e., the top-level directory) by running `cd /`.
7. Run the following command:<pre><code class="language-bash">docker run -it --name deepks-container --network host --hostname deepks-container benndrucker/deepks</code></pre>
You should see this prompt: <code class = "inline-bash-output">(base) //deepks-container// [/] ▷ </code>. You are now in the DeepKS container. You must run DeepKS commands from here.
Run the following command:

***Note: You will have `sudo` privileges inside the Docker container (!) by virtue of passing `--fakeroot`. If you ever need to install programs, for example, this means you can do so inside the container.***

***Note: The following steps are run from <u> inside the Docker container</u>. See the steps above to start the Docker container.***

# Trying Examples

Because of various Python specifications, `DeepKS.examples` — the examples submodule — must be run as a module from _outside_ the `/DeepKS` directory. Hence, before running examples, ensure you are in the top-level directory (which contains a symbolic link (an alias) to `DeepKS`).  The examples submodule provides sample api calls to give the user a sense of how to use the tool. The full API specification is listed in the next section.

Use the following command to actually start the examples:

<pre><code class="language-bash">python3 -m DeepKS.examples</code></pre>
<p>This will run a few examples of DeepKS. You should see the following output (with more lines at the end):</p>
<iframe src='example_output.html', height = '750px', width = '100%' id='exiframe'></iframe>

# Full `DeepKS` Specification
Below is a link to view the full code specification for `DeepKS`. This contains documentation for the functions in this package.
<div style='width:100%;justify-content: center;align-items: center; display: flex;'>
<div style='justify-content: center;align-items: center; display: flex;'>
<div style='text-align: center'><a href='api_pydoctor_docs/index.html' class='no-underline'><div class='fancy-link'>➥</div></a></div>
</div>
</div>

# Using Command Line API
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
                                 [--kin-info <kinase info file>] [--site-info <site info file>]
                                 [--cartesian-product]
                                 [-p {inorder,dictionary,in_order_json,dictionary_json,csv,sqlite}]
                                 [--suppress-seqs-in-output] [-v]
                                 [--pre-trained-nn <pre-trained neural network file>]
                                 [--pre-trained-gc <pre-trained group classifier file>]
                                 [--pre-trained-msc <pre-trained multi-stage classifier file>]
                                 [--group-on <group on>] [--device <device>] [--scores]
                                 [--normalize-scores] [--groups] [--bypass-group-classifier]
                                 [--dry-run] [--convert-raw-to-prob]
```
- Anything in square brackets is optional and has default values. To view what these flags refer to (and their default values), run `python -m DeepKS.api.main --help`.
- For each instance of round parentheses, you must provide one of the options between "`|`". 
- Curly braces show available options for a flag.
- That is, as a minimal example, you may run `python -m DeepKS.api.main -kf my_kinase_sequences.txt -sf my_site_sequences.txt`.
- You might also run `python3 -m DeepKS.api.main -kf my_kinase_sequences.txt -sf my_site_sequences.txt --kin-info my_kinase_info.txt --site-info my_site_info.txt --cartesian-product -p in_order_json -v --pre_trained_nn my_pre_trained_nn.pt --pre_trained_gc my_pre_trained_gc.pt --device cuda:0 --scores --normalize-scores --groups --dry-run`.
- A note on `--lower-logging-level` and `--upper-logging-level`: Each type of log statment in the program (e.g., Status, Info, Warning, Error) is associated with an integer. The integer value increases with the importance/severity of the log statement. Below is a table of integers, their associated log statements, and their usages.


|Log Type | Integer | Log Type Description |
|---------|---------|----------------------|
|Debug    |   5     | Log debugging statements.|
|Vanishing Status |9| Log a status that is overwritten by the next logging statement.     |
|Status   |   10    | Log a status that is not overwritten by the next logging statement. |
|Progress |   15    | Log progress bars at and above this level                           |
|Info     |   20    | Log information not related to program progress.                    |
|Train Info|  21    | Log information about performance in the training process.          |
|Validation Info| 23| Log information about performance in the validation process.        |
|Res Info |   25    | Log information about performance in the testing process.           |
|Warning  |   30    | Log warnings                                                        |
|User Error|  35    | Log errors that are caused and/or partially anticipated by the user.|
|Error    |   45    | Log errors that are totally unexpected                              |

The following is an example of how the above log types are presented:


***Note: If using CUDA, it may be helpful to run `nvidia-smi` to see which GPUs is being used extensively. DeepKS can automatically scale for the available hardware, but it will run much faster if it is run on a GPU with no other concurrent processes.***

Here are some more examples of how to run the program (make sure to be in the top-level `/` directory):

```bash
python -m DeepKS.api.main -kf my/kinase/sequences.txt -sf my/site/sequences.txt -p in_order_json -v True --device cuda:4

python -m DeepKS.api.main -k KINASE_SEQ_1,KINASE_SEQ_2,KINASE_SEQ_3 -s SITE_SEQ_1,SITE_SEQ_2,SITE_SEQ_3 -p dictionary

python -m DeepKS.api.main -kf my/kinase/sequences.txt -s SITE_SEQ_1,SITE_SEQ_2,SITE_SEQ_3 -p inorder -v False

python -m DeepKS.api.main -kf my/kinase/sequences.txt -sf my/site/sequences.txt --dry-run
```
***Note: The example files above are for example purposes only and don't actually exist.***

### Adding to new code and importing
#### VS Code Integration
To make adding DeepKS to an external codebase, we recommend using VS Code. To do this, follow these steps:
1. Open VS Code.
2. Install the following extensions by searching for them:
   - Dev Containers
   - Remote - Tunnels
   - Remote Explorer
   - Remote - Tunnels
   - WSL (if using WSL)
3. Go back to your terminal, and make sure you're inside the `benndrucker/deepks:latest` Docker container. Then run `/usr/share/code/bin/code-tunnel tunnel` and follow the instructions.

#### Importing
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


## Running tests
It may be useful to run the tests to make sure everything is working properly (especially if the user modifies the DeepKS API). To do this run — from the top-level directory — `python3 -m unittest -fvb DeepKS.tests.test`. If all tests pass, and you want to see code coverage, you can run `coverage run -m unittest discover -fvb` and then `coverage -m report`.

# File Explainer
Below, you will find a scrollable tree of files in this repository and their descriptions. Boldfaced nodes represent directories.
<div><iframe src="tree.html" width=100% height=500px style="overscroll-behavior:contain;"></iframe></div>

# Troubleshooting
Here are some general troubleshooting guidelines:
- Generally, errors will be presented in the following format:

<pre>
<span style="color:purple;">Full Traceback:</span>
<span style="color:purple;">
DeepKS/tools/informative_tb.py:121, in &lt;module&gt;
    fn_a()
</span><span style="color:purple;">
  DeepKS/tools/informative_tb.py:106, in fn_a
      fn_b()
</span><span style="color:purple;">
    DeepKS/tools/informative_tb.py:110, in fn_b
        fn_c()
</span><span style="color:purple;">
      DeepKS/tools/informative_tb.py:114, in fn_c
          fn_d()
</span><span style="color:purple;">
        DeepKS/tools/informative_tb.py:118, in fn_d
    ---&gt;    raise RuntimeError(&quot;This is a test exception!&quot;)
</span>


<span style="color:red;">Error: Something went wrong! Error message(s) below. Full traceback above.
</span>
<span style="color:red;">==========================================================================================</span>
<span style="color:purple;">  * Error Type: RuntimeError (Description: Raised when an error is detected that doesn't
        fall in any of the other categories. The associated value is a string indicating
        what precisely went wrong.)</span>
<span style="color:purple;">  * Error Message: This is a test exception!</span>
<span style="color:purple;">  * Error Location: DeepKS/tools/informative_tb.py:121</span>
<span style="color:purple;">  * Error Function: &lt;module&gt;</span>
<span style="color:red;">==========================================================================================</span>

</pre>
- Interpreting this:
  - At the top is the full traceback, which details the origin of the error.
  - At the bottom is an error summary, which includes the error type, message, location, and function. There is also a description of the error type. This is a generic description pulled from the Python documentation. The error message may be more useful and specific.
  - If the user cannot determine the cause of the error and fix it, they can open an "error" [issue on the GitHub repository](https://github.com/Ben-Drucker/DeepKS/issues/new/choose). Click "Get started" to the right of "user-error". Then, add a title, copy and paste the required information (based on the issue template), and click "Submit new issue".
  - If no errors were raised, but the output was unexpected, they can open an "unexpected output" issue [on the GitHub repository](https://github.com/Ben-Drucker/DeepKS/issues/new/choose). Click "Get started" to the right of "user-unexpected output". Then, add a title, copy, paste, and add the required information (based on the issue template), and click "Submit new issue".
  - For all other issues, on [the issues page](https://github.com/Ben-Drucker/DeepKS/issues/new/choose), one can select "Don’t see your issue here? Open a blank issue." at the bottom. (But make sure to ensure the issue doesn't fit into the other two categories first.)

# Reproducing Everything From Scratch
## Preprocessing and Data Collection
All the steps are packaged in the module `DeepKS.data.preprocessing.main`. This will run through all the necessary steps, download the necessary data, and put it in the appropriate directories. It will also report what it is doing as it runs.
## Training
The python training scripts contain command line interfaces. However, to make running easier, one can use the bash scripts in the `models` directory. The bash scripts are simply wrappers around the python scripts. The bash scripts are the recommended way to run the training scripts.
1. Run `bash models/train_group_classifier.sh` to train and save the group classifier.
2. Run `bash models/train_individual_classifiers.sh` to train and save the individual group-based neural network classifiers. This should be done on a GPU, if possible. If not, it will take a long time. However, the script automatically detects the amount of memory in the system so it won't get overloaded and/or crash.
## Evaluating
TODO: finish this section
## Creating Evaluation Diagrams
The evaluation diagram creating functions are located in the module `DeepKS.models.DeepKS_evaluation`. TODO: finish this section
## Creating Other Diagrams
Other supporting diagrams can be made from the submodules of `DeepKS.images`. TODO: finish this section
</div>
</div>
</div>
<script>
document.querySelector("head > title:nth-child(2)").innerHTML = "DeepKS Manual";
</script>