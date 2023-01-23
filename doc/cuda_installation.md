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
# Explanation
Since docker/cuda is not supported on Windows, we will use WSL (Windows Subsystem for Linux) to run docker and cuda. To accomplish this, you must download from the DeepKS repository the following file: [cuda_wsl_installer.sh](https://gitlab.com/Ben-Drucker/deepks/-/raw/main/build/cuda_wsl_installer.sh?inline=false). For consistency with the instructions below, please save this file in your Downloads folder.

# WSL Setup
***Note: It is recommended to run each command, one line at a time so that you can see the output of each command and make sure each worked.***

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
sudo cd ~ && sudo chmod +x ./cuda_wsl_installer.sh  && sudo ./cuda_wsl_installer.sh
```
It will ask for your password if you haven't already used `sudo` in the current session.

When this completes, you need to start the Docker Daemon. The Daemon is a program that runs in the background and serves Docker containers. To start the Daemon, run

```{bash}
sudo service docker start
```

To ensure this starts each time you open WSL, run

```{bash}
sudo update-rc.d docker defaults # FIXME: This doesn't work
```
(This only needs to be done once.)

# Finishing
To check that all this worked, run 

```{bash}
sudo docker run hello-world
```

You should get the following (possibly with a different hash and more text):

<pre class = "bash-output bash-output">
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
</pre>

Then, you may proceed back to the [main help page, §Terminolgy](https://ben-drucker.gitlab.io/deepks/#terminology)

# Troubleshooting
### If you get a permissions error like the following:

<pre class = "bash-output bash-output">
docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/create": dial unix /var/run/docker.sock: connect: permission denied.
See 'docker run --help'.
</pre>

you probably need to run `docker` with `sudo`. (I.e., `sudo docker ...`)

### If you get a Docker Daemon error like the following:

<pre class = "bash-output bash-output">
docker: Error response from daemon: dial unix docker.raw.sock: connect: connection refused.
</pre>

or 

<pre class = "bash-output bash-output">
docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?.
See 'docker run --help'.
</pre>

you probably need to start the Docker Daemon. (See above.)