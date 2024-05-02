# planutils

General library for setting up linux-based environments for developing, running, and evaluating planners.

## Installation

### Docker (Recommended)
1. Install Docker if you haven't already (if you're on Ubuntu, you can follow Steps 1-5 on [this](https://www.tecmint.com/install-docker-and-run-docker-containers-in-ubuntu/) link)
2. Once you're able to run the `hello-world` Docker example correctly, go ahead and install the `planutils` Dockerfile (you may need to use `sudo` or the equivalent if you're running into permission issues)
    ```bash
    docker pull aiplanning/planutils
    ```
3. Now, if you use the `sudo docker images` command, you should be able to see all the `aiplanning/planutils` container listed
4. You're done! Now, you can pull up a bash terminal - wherein you can run all of planutils' commands - within the container like so:
    ```bash
    docker run --privileged -it aiplanning/planutils bash
    ```
5. To allow the planners to access domain and problem files you might have locally, you need to use a docker bind-mount like so:
    ```bash
    sudo docker run --privileged -v /home/nishanth/Documents/planutils_stuff:/mnt/ -it aiplanning/planutils bash
    ```
    Here, the `/home/planutils_stuff` folder of your local machine will be connected to the `/mnt` folder of your Docker install. You can now read from and write to this local folder.


### pip
```bash
pip3 install planutils
```

Note that most of the planners included with planutils require a modern version of [Singularity](https://sylabs.io/docs/), which can be rather tricky to install depending on your OS and development environment. For this reason, the recommended installation method is using Docker (as described above)

## Usage

### Example of currently working functionality

```bash
$ lama domain.pddl problem.pddl

Package not installed!
  Download & install? [Y/n] y

About to install the following packages: downward (36M), lama (20K)
  Proceed? [Y/n] y
Installing downward...
INFO:    Downloading shub image
 35.88 MiB / 35.88 MiB [=======================================] 100.00% 3.99 MiB/s 8s
Finished installing downward (size: 36M)
Installing lama...
Finished installing lama (size: 20K)
Successfully installed lama!

Original command: lama
  Re-run command? [Y/n] y

Parsing...
$
```

### Example of upcoming functionality

```bash
$ planutils install ipc-2018
Installing planners
This will require 3Gb of storage. Proceed? [Y/n]
Fetching all of the planners from IPC-2018 for use on the command line...

$ planutils install server-environment
Setting up a webserver to call the installed planners...

$ planutils install development-environment
Installing common dependencies for building planners...
Installing common planning libraries...

$ planutils install planning-domains
Installing the command-line utilities...
Installing the python library...
Fetching default benchmarks...

$ planutils setup-evaluation configuration.json
Installing Lab...
Configuring Lab...
Ready!
Run eval.py to evaluate

$
```

### Docker

The included Docker file will come with planutils pre-installed. Note that in order to
run a number of the planners (all those that are based on singularity), you will need
to run the docker with the `--privileged` option.
