# Neural Network Models Compilation and Execution Guide

This guide provides instructions on how to compile and execute four different neural network models, each paired with a specific supplementary file. Below are the details for each pair of files.

## Model CPU Native (`model_cpu_native.c` and `supp_cpu_native.h`)

### Compilation
To compile `model_cpu_native.c`, use the following command:

```bash
gcc -O3 -o model_cpu_native model_cpu_native.c -lm
```

### Execution
To execute the compiled program, use the following command:

```bash
./model_cpu_native <hidden_layers> <nodes_per_layer> <epochs> <batch_size> <learning_rate>
```

## Model CPU BLAS (`model_cpu_blas.c` and `supp_cpu_blas.h`)

### Compilation
To compile `model_cpu_blas.c`, use the following command:

```bash
gcc -O3 -o model_cpu_blas model_cpu_blas.c -lm -lcblas
```

### Execution
To execute the compiled program, use the following command:

```bash
./model_cpu_blas <hidden_layers> <nodes_per_layer> <epochs> <batch_size> <learning_rate>
```

## Model GPU Native (`model_gpu_native.cu` and `supp_gpu_native.h`)

### Compilation
Before compiling, ensure CUDA is loaded. To compile `model_gpu_native.cu`, use the following command:

```bash
nvcc -O3 -arch=sm_70 -o model_gpu_native model_gpu_native.cu
```

### Execution
To execute the compiled program, use the following command:

```bash
./model_gpu_native <hidden_layers> <nodes_per_layer> <epochs> <batch_size> <learning_rate>
```

## Model GPU CuBLAS (`model_gpu_cublas.cu` and `supp_gpu_cublas.h`)

### Compilation
Before compiling, ensure CUDA is loaded. To compile `model_gpu_cublas.cu`, use the following command:

```bash
nvcc -O3 -arch=sm_70 -o model_gpu_cublas model_gpu_cublas.cu -lcublas
```

### Execution
To execute the compiled program, use the following command:

```bash
./model_gpu_cublas <hidden_layers> <nodes_per_layer> <epochs> <batch_size> <learning_rate>
```

Replace `<hidden_layers>`, `<nodes_per_layer>`, `<epochs>`, `<batch_size>`, and `<learning_rate>` with your desired values when executing the programs.
