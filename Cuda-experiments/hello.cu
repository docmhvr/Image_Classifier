#include <iostream>

#define CUDA_CHECK_ERROR(err) if (err != cudaSuccess) { \
  printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
  return -1; \
}

__global__ void myKernel() {
    printf("Hello, World from GPU!\n");
}

int main() {
    myKernel<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();  // Check kernel launch
    CUDA_CHECK_ERROR(err);

    err = cudaDeviceSynchronize();  // Check runtime errors
    CUDA_CHECK_ERROR(err);
    return 0;
}
