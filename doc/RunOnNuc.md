# How to run AIMM on an Intel NUC (Next Unit of Computing) without AVX (Advanced Vector Extensions)

## Introduction
We struggled to run the model on an Intel NUC with a Intel Pentium Silver processor. The attempt resulted in a core dump with the message 'Instruction not supported'. A bit of research showed that our Pentium comes without AVX support. However, Tensorflow (one of the dependencies of our application) requires by default AVX support. The following article shows a workarround.

## Step 0 - Install Ubuntu Server and get the source code

1. Install Ubuntu Server 22.04 (Jammy) on the Intel NUC
2. Use Git to get the AIMM source code

## Step 1 - Identify your Tensorflow version
 
1. Install the requirements by using the following command: `pip3 install -r requirements.txt`
2. Identify the Tensorflow version that was installed, e.g., `2.13`
3. Uninstall Tensorflow again: `pip3 uninstall tensorflow`

## Step 2 - Build Tensorflow from source

1. Follow the build instructions [here](https://www.tensorflow.org/install/source)
2. Use Git to check the correct version branch out, e.g. `r2.13`
3. You will need to install the tool `bazel` and a bunch of other dependencies
4. Get a coffee and wait for a bunch of hours.

Here are the build instructions that I used after having set up my build environment:

* Build the library without AVX:

```
bazel build --config=native_arch_linux --local_ram_resources=2048 //tensorflow/tools/pip_package:build_pip_package
```

* Build the Python package:

```
/tmp/tensorflow_pkg/
pip3 install tensorflow-2.13.0-cp310-cp310-linux_x86_64.whl 
```
## Step 3 - Run the application

1. Try to restart the application again `python3 app.py`
2. There shouldn't be a core dump anymore



