#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


#define FLOAT4(value)  *(float4*)(&(value))

template <typename T> 
inline T CeilDiv(const T& a, const T& b) {
    return (a + b - 1) / b;
}

const size_t N = 32ULL*1024ULL*1024ULL;
//const size_t N = 640*256; // data size, 163840
const int BLOCK_SIZE = 256;
// naive atomic reduction kernel
__global__ void atomic_red(const float *gdata, float *out){
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < N) atomicAdd(out, gdata[idx]);
}

//block_reduce
__global__ void reduce(float *gdata, float *out){
     __shared__ float sdata[BLOCK_SIZE];//增大BLOCK_SIZE会使得同一时间驻留在SM里的block变少
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < N) {  // grid stride loop to load data
        sdata[tid] += gdata[idx];
        idx += gridDim.x*blockDim.x;
        }

     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)  // parallel sweep reduction
            sdata[tid] += sdata[tid + s];
        }
     if (tid == 0) out[blockIdx.x] = sdata[0];
  }

//sweep style parallel reduction,规约求和，使用shared memory
 __global__ void reduce_a(float *gdata, float *out){
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < N) {  // grid stride loop to load data
         sdata[tid] += gdata[idx];
         idx += gridDim.x*blockDim.x;
      }

     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        }
     if (tid == 0) atomicAdd(out, sdata[0]);
  }


__global__ void reduce_ws(float *gdata, float *out){
     __shared__ float sdata[32];//最多就32个warp，因为线程数最多是1024
     int tid = threadIdx.x;
     int idx = threadIdx.x+blockDim.x*blockIdx.x;

     float val = 0.0f;//这是每个线程独有的本地变量
     unsigned mask = 0xFFFFFFFFU;

     int lane = threadIdx.x % warpSize;//0~31
     int warpID = threadIdx.x / warpSize;

     while (idx < N) {  // grid stride loop to load 
        val += gdata[idx];
        idx += gridDim.x*blockDim.x;  
      }

 // 1st warp-shuffle reduction
   #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) 
       val += __shfl_xor_sync(mask, val, offset);

    if (lane == 0) sdata[warpID] = val; //warp内部求和, 每个线程都把warp内部求和的结果放在自己本地的val变量上，但是只有lane==0的val被保存进shared_memory[warpID]里
   __syncthreads();

    if (warpID == 0)
    {
 // reload val from shared mem if warp existed
       val = (tid < blockDim.x/warpSize)?sdata[lane]:0;//为什么用lane而不是warpID去索引？ 因为warpID是0才会进来
       //tid 小于8，那么lane也是0~7之间，可以用来索引sdata
 // final warp-shuffle reduction
       for (int offset = warpSize/2; offset > 0; offset >>= 1)  
          val += __shfl_down_sync(mask, val, offset);

       if(tid == 0) atomicAdd(out, val);//将所有block内部求和的结果再一次汇总
     }
  }


__global__ void reduce_ws_float4(float *gdata, float *out){

   __shared__ float sdata[32];//最多就32个warp，因为线程数最多是1024
   int tid = threadIdx.x;
   int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

   float val = 0.0f;//这是每个线程独有的本地变量
   unsigned mask = 0xFFFFFFFFU;

   int lane = threadIdx.x % warpSize;//0~31
   int warpID = threadIdx.x / warpSize;

   while (idx < N) {  // grid stride loop to load 
      float4 tmp_input = FLOAT4(gdata[idx]); 
      val += tmp_input.x;
      val += tmp_input.y;
      val += tmp_input.z;
      val += tmp_input.w;
      idx += (gridDim.x*blockDim.x)*4;  
   }

   // 1st warp-shuffle reduction
   #pragma unroll
   for (int offset = warpSize/2; offset > 0; offset >>= 1)
      val += __shfl_xor_sync(mask, val, offset);
       
   if (lane == 0) sdata[warpID] = val; //warp内部求和, 每个线程都把warp内部求和的结果放在自己本地的val变量上，但是只有lane==0的val被保存进shared_memory[warpID]里
   __syncthreads();

   if(warpID == 0){
      val = (tid < blockDim.x/warpSize) ? sdata[lane] : 0;
      for (int offset = warpSize/2; offset > 0; offset >>= 1)  
          val += __shfl_down_sync(mask, val, offset);

      if(tid == 0) atomicAdd(out, val);
   }
}


int main(){

  float *h_A, *h_sum, *d_A, *d_sum;
  h_A = new float[N];  // allocate space for data in host memory
  h_sum = new float;
  for (int i = 0; i < N; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, N*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sum, sizeof(float));  // allocate device space for sum
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete


  atomic_red<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("atomic reduction kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

  //cuda processing sequence step 3 is complete
  cudaCheckErrors("atomic reduction kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("atomic sum reduction incorrect!\n"); /*return -1;*/}
  printf("atomic sum reduction correct!\n");
  const int blocks = 640;
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete


  reduce_a<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction w/atomic kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction w/atomic sum incorrect!\n"); /*return -1;*/}
  printf("reduction w/atomic sum correct!\n");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete


  //reduce_ws<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  reduce_ws_float4<<<blocks/4, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction warp shuffle kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction warp shuffle kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction warp shuffle sum incorrect!\n"); /*return -1;*/}
  printf("reduction warp shuffle sum correct!\n");
  return 0;
}
  
