{::options parse_block_html="true" /}
### WSL Setup
***Note: It is recommended to run each command, one line at a time so that you can see the output of each command and make sure each worked.***

#### Since docker/cuda is not supported on Windows, we will use WSL (Windows Subsystem for Linux) to run docker and cuda. To accomplish this, you must download from the DeepKS repository the following file: [cuda_wsl_installer.sh](https://gitlab.com/Ben-Drucker/deepks/-/raw/main/build/cuda_wsl_installer.sh?inline=false). For consistency, it is please save this file in your Downloads folder.

To install WSL, open Terminal (or powershell) as an administrator and run the following commands:
```{powershell}
wsl.exe --install
wsl.exe --update
wsl.exe --install -d Ubuntu-20.04
```
Then follow instructions to create a username and password. (Recommended username: pnnl) To use WSL in the future, simply run

```{powershell}
wsl
```
From now on, use the WSL terminal to run the following commands.

To install the necessary CUDA Toolkit and docker, first copy the installation scripts to your WSL home directory. To do this, run the following commands in the WSL terminal:

```{bash}
cp /mnt/c/Users/<your_username>/Downloads/cuda_wsl_installer.sh ~/

```
making sure to replace `<your_username>` with your actual username. Then run 

```{bash}
sudo chmod +x ./cuda_wsl_installer.sh  && sudo ./cuda_wsl_installer.sh
```
It will ask for your password if you haven't already used `sudo` in the current session.

To check this worked, run 

```{bash}
docker run hello-world
```

You should get the following (possibly with a different hash and more text):

<span style="color:green">
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
0e03bdcc26d7: Pull complete
Digest: sha256:8b5a7d9e0e178f2f37a820e3f795c19c4c2522b3f282a2f9d2a8b626cf6d8e0a
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
1. The Docker client contacted the Docker daemon.
2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
$ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
https://hub.docker.com/

For more examples and ideas, visit:
https://docs.docker.com/get-started/
</span>

Then, 
