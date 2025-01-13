#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include<assert.h>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define FLOAT4(value)  *(float4*)(&(value))

#define checkCudaErrors(func)               \
{                                   \
    cudaError_t e = (func);         \
    if(e != cudaSuccess)                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));        \
}

//nvcc -o relu relu.cu && ./relu
//sigmoid<<<CeilDiv(N, block_size), block_size>>>(d_A, d_B, N)
//a: Nx1, b: Nx1, c: Nx1, y = relu(x)
__global__ void relu(float* x, float* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

__global__ void relu_float4(float* x, float* y, int N){

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if(idx < N){
        float4 tmp_x = FLOAT4(x[idx]);
        float4 tmp_y;
        tmp_y.x = fmaxf(0.0f, tmp_x.x);
        tmp_y.y = fmaxf(0.0f, tmp_x.y);
        tmp_y.z = fmaxf(0.0f, tmp_x.z);
        tmp_y.w = fmaxf(0.0f, tmp_x.w);
        FLOAT4(y[idx]) = tmp_y;
    }
}


template <typename T> 
inline T CeilDiv(const T& a, const T& b) {
    return (a + b - 1) / b;
}


int main(){

    size_t block_size = 128;
    size_t N = 1 * 1024;
    size_t bytes_A = sizeof(float) * N;
    size_t bytes_B = sizeof(float) * N;

    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);

    for( int i = 0; i < N; i++ ){
        h_A[i] = (i / 666) * ((i % 2 == 0) ? 1: -1);
    }

    float* d_A;
    float* d_B;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msec = 0;
    int iteration = 1000;
    checkCudaErrors(cudaEventRecord(start));
    for(int i = 0; i < iteration; i++)
    {
        //relu<<<CeilDiv(N, block_size), block_size>>>(d_A, d_B, N);
        //relu_float4<<<CeilDiv(N, block_size), block_size/4>>>(d_A, d_B, N);
        relu_float4<<<CeilDiv(N/4, block_size), block_size>>>(d_A, d_B, N);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
    printf("relu takes %.3f msec\n", msec/iteration);
    checkCudaErrors(cudaMemcpy(h_B, d_B, bytes_B, cudaMemcpyDeviceToHost));

    for(int i = 0; i < N; i++){
        double err = fabs(h_B[i] - fmaxf(0.0f, h_A[i]));

        if(err > 1.e-6) {
            printf("wrong answer!\n");
            break;
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);

    free(h_A);
    free(h_B);

    return 0;
}