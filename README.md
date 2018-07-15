

- [PUMPS 2018](#pumps-2018)
  - [Setup](#setup)
  - [Labs](#labs)
      - [Run Options](#run-options)
  - [Project Build Specification](#project-build-specification)
  - [Profiling](#profiling)
  - [Utility Functions](#utility-functions)
    - [Verifying the Results](#verifying-the-results)
    - [How to Time](#how-to-time)
    - [Checking Errors](#checking-errors)
    - [Enabling Debug builds](#enabling-debug-builds)
  - [Offline Development](#offline-development)
  - [Issues](#issues)
  - [License](#license)

# PUMPS 2018

## Setup

Clone this repository to get the project folder.

    git clone https://github.com/illinois-impact/pumps.git

Download the rai binary for your platform.
You will probably use it for development, and definitely use it for submission.


| Operating System | Architecture | Stable Version (0.2.57) Link                                                             | Beta Version (0.2.57) Link                                                              |
| ---------------- | ------------ | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Linux            | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.57/linux-amd64.tar.gz)   | [URL](https://github.com/rai-project/rai/releases/download/latest/linux-amd64.tar.gz)   |
| OSX/Darwin       | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.57/darwin-amd64.tar.gz)  | [URL](https://github.com/rai-project/rai/releases/download/latest/darwin-amd64.tar.gz)  |
| Windows          | amd64        | [URL](https://github.com/rai-project/rai/releases/download/v0.2.57/windows-amd64.tar.gz) | [URL](https://github.com/rai-project/rai/releases/download/latest/windows-amd64.tar.gz) |

You should have received a `.rai_profile` file by email.
Put that file in `~/.rai_profile` (Linux/macOS) or `%HOME%/.rai_profile` (Windows).
Your `.rai_profile` should look something like this (indented with tabs!)

    profile:
        firstname: <your-given-name>
        lastname: <your-surname>
        username: <your-username>
        email: <your-institution-email>
        access_key: <your-access-key>
        secret_key: <your-secret-key>
        affiliation: <your-affiliation>

Some more info is available on the [Client Documentation Page](https://github.com/rai-project/rai).


rai will run

## Labs

One or more labs are going to be assigned for each day.

- [Device Query](labs/device_query)
- [Scatter](labs/scatter)
- [Gather](labs/gather)
- [Binning](labs/binning)
- [Basic Convolution Layer](labs/basic_conv)

The main code of each lab is in the `main.cu`, which is the file you will be editing. Helper code that's specific to the lab is in the `helper.hpp` file and the common code across the labs in the `common` folder. You are free to add/delete/rename files but you need to make the appropriate changes to the `CMakeLists.txt` file.

To run any lab you `cd` into that directory, `cd labs/device_query` for example, and run `rai -p .` .
From a user's point a view when the client runs as if it was local.
In reality, the local directory specified by `-p` gets uploaded to the server and extracted into the `/src` directory on the server. The server then executes the build commands from the `rai_build.yml` specification within the `/build` directory. Once the commands have been run, or there is an error, a zipped version of that `/build` directory is available from the server for download.

The server limits the task time to be an hour with a maximum of 8GB of memory being used within a session. The output `/build` directory is only available to be downloaded from the server for a short amount of time. Networking is also disabled on the execution server. Contact the teaching assistants if this is an issue.

#### Run Options

      -c, --color         Toggle color output.
      -d, --debug         Toggle debug mode.
      -p, --path string   Path to the directory you wish to submit. Defaults to the current working directory. (default "current working directory")
      -v, --verbose       Toggle verbose mode.

On Windows, it might be useful to disable the colored output. You can do that by using the `-c=false` option

_NOTE:_ You may need to use the absolute path if submitting from windows.



## Project Build Specification

The `rai_build.yml` must exist in your project directory. In some cases, you may not be able to execute certain builtin bash commands, in this scenario the current workaround is to create a bash file and insert the commands you need to run. You can then execute the bash script within `rai_build.yml`.

The `rai_build.yml` is written as a [Yaml](http://yaml.org/) ([Spec](http://www.yaml.org/spec/1.2/spec.html)) file and has the following structure.

```yaml
rai:
  version: 0.2 # this is required
  image: nimbix/ubuntu-cuda-ppc64le:latest # nimbix/ubuntu-cuda-ppc64le:latest is a docker image
                                           # You can specify any image found on dockerhub
resources:
  cpu:
    architecture: ppc64le
  gpu:
    architecture: pascal
    count: 1 # tell the system that you're using a gpu
  network: false
commands:
  build:
    - echo "Building project"
    # Use CMake to generate the build files. Remember that your directory gets uploaded to /src
    - cmake /src
    # Run the make file to compile the project.
    - make
    # here we break the long command into multiple lines. The Yaml
    # format supports this using a block-strip command. See
    # http://stackoverflow.com/a/21699210/3543720 for info
    - >-
      ./mybinary -i input1,input2 -o output
```

Syntax errors will be reported, and the job will not be executed. You can check if your file is in a valid yaml format by using tools such as [Yaml Validator](http://codebeautify.org/yaml-validator).


## Profiling

Profiling can be performed using `nvprof`. Place the following build commands in your `rai-build.yml` file

```yaml
    - >-
      nvprof --cpu-profiling on --export-profile timeline.nvprof --
      ./mybinary -i input1,input2 -o output
    - >-
      nvprof --cpu-profiling on --export-profile analysis.nvprof --analysis-metrics --
      ./mybinary -i input1,input2 -o output
```

You could change the input and test datasets. This will output two files `timeline.nvprof` and `analysis.nvprof` which can be viewed using the `nvvp` tool (by performing a `file>import`). You will have to install the nvvp viewer on your machine to view these files.

_NOTE:_ `nvvp` will only show performance metrics for GPU invocations, so it may not show any analysis when you only have serial code.

You will need to install the nvprof viewer for the CUDA website and the nvprof GUI can be run without CUDA on your machine.


## Utility Functions

We provide a some helper utility functions in the `common/utils.hpp` file.

### Verifying the Results

Each lab contains the code to compute the golden (true) solution of the lab. We use [Catch2](https://github.com/catchorg/Catch2) to perform tests to verify the results are accurate within the error tollerance. You can read the [Catch2 tutorial](https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md#top) if you are interested in how this works.

Subsets of the test cases can be run by executing a subset of the tests. We recomend running the first lab with `-h` option to understand what you can perform, but the rough idea is if you want to run a specific section (say `[inputSize:1024]`) then you pass `-c "[inputSize:1024]"` to the lab.

### How to Time

In `common/utils.hpp` a function called `timer_start/timer_stop` which allows you to get the current time at a high resolution. To measure the overhead of a function `f(args...)`, the pattern to use is:

```{.cpp}
timer_start(/* msg */ "calling the f(args...) function");
f(args...);
timer_stop();
```

This will print the time as the code is running

### Checking Errors

To check and throw CUDA errors, use the THROW_IF_ERROR function. This throws an error when a CUDA error is detected which you can catch if you need special handling of the error.

```{.cpp}
THROW_IF_ERROR(cudaMalloc((void **)&deviceW, wByteCount));
```


### Enabling Debug builds

Within the `rai_build.yml` environment, run `cmake -DCMAKE_BUILD_TYPE=Debug /src` this will enable debugging mode. You may also want to pass the `-g -pg -lineinfo` option to the compiler.

## Offline Development

You can use the docker image and or install CMake within a CUDA envrionment. Then run `cmake [lab]` and then `make`. We do not using your own machine, and we will not be debugging your machine/installation setup.

## Issues


Please use the [Github issue manager] to report any issues or suggestions.

Include the outputs of

```bash
rai version
```

as well as the output of

```bash
rai buildtime
```

In your bug report. You can also invoke the `rai` command with verbose and debug outputs using

```bash
rai --verbose --debug
```

[github issue manager]: https://github.com/illinois-impact/pumps/issues

## License

NCSA/UIUC © [Abdul Dakkak](http://impact.crhc.illinois.edu/Content_Page.aspx?student_pg=Default-dakkak)

