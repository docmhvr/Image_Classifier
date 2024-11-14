#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// define M,K,N,BLOCK_SIZE as 1024,512,2048,32

// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n){
    // pass
}

// CUDA kernel for matrix multiplication
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n){
    // pass
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols){
    // pass
}

double get_time() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time_in_seconds = std::chrono::duration<double>(now.time_since_epoch()).count();
    return time_in_seconds;
}

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}