#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <limits.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cuda.h>

#define INF 99999

// static int bs = 64;
#define BLOCK_DIM 16
#define bs 16
#define BLOCK_SIZE 64
#define HALF_BLOCK_FACTOR 64
// const int num_threads = 16; // 調整為您需要的數量
__forceinline__
__device__ void block_calc(int* C, int* A, int* B, int bi, int bj) {
  for (int k = 0; k < bs; k++) {
    int sum = A[bi*bs + k] + B[k*bs + bj];
    if (C[bi*bs + bj] > sum) {
      C[bi*bs + bj] = sum;
    }
    __syncthreads();
  }
  
}

__global__ void floyd_warshall_block_kernel_phase1(int n, int k, int* graph) {
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  // Transfer to temp shared arrays
  C[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();
  
  block_calc(C, C, C, bi, bj);

  __syncthreads();

  // Transfer back to graph
  graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];

}


__global__ void floyd_warshall_block_kernel_phase2(int n, int k, int* graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int i = blockIdx.x;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k) return;

  __shared__ int A[BLOCK_DIM * BLOCK_DIM];
  __shared__ int B[BLOCK_DIM * BLOCK_DIM];
  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  C[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];
  B[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, C, B, bi, bj);

  __syncthreads();

  graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];

  // Phase 2 1/2

  C[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + i*BLOCK_DIM + bi*n + bj];
  A[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, A, C, bi, bj);

  __syncthreads();

  // Block C is the only one that could be changed
  graph[k*BLOCK_DIM*n + i*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];
}


__global__ void floyd_warshall_block_kernel_phase3(int n, int k, int* graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int j = blockIdx.x;
  const unsigned int i = blockIdx.y;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k && j == k) return;
  __shared__ int A[BLOCK_DIM * BLOCK_DIM];
  __shared__ int B[BLOCK_DIM * BLOCK_DIM];
  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  C[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj];
  A[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];
  B[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, A, B, bi, bj);

  __syncthreads();

  graph[i*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];
}

__global__ void block_fw_p1_kernel(int* dist1, int k, int num_vertices)
{
    __shared__ int shared_block[bs*bs];
    __syncthreads();
    
    int x_shift = k*bs;
    int y_shift = k*bs + threadIdx.y * num_vertices + threadIdx.x;  //
    shared_block[threadIdx.y*bs + threadIdx.x] = dist1[x_shift*num_vertices + y_shift];
    __syncthreads();
    block_calc(shared_block,shared_block,shared_block,threadIdx.y,threadIdx.x);
    // int tmp;
    // for(int kk=0;kk<bs;kk++){
    //     tmp = shared_block[threadIdx.x*bs + kk] + shared_block[kk*bs + threadIdx.y];  
    //     if(tmp < shared_block[threadIdx.x*bs + threadIdx.y])
    //         shared_block[threadIdx.x*bs + threadIdx.y] = tmp;
    //     __syncthreads();
    // }
    __syncthreads();
    dist1[x_shift*num_vertices + y_shift] = shared_block[threadIdx.y*bs + threadIdx.x];
}

__global__ void block_fw_p2_kernel(int* dist1, int k, int num_vertices)
{
    if(blockDim.x == k)
        return;
    __shared__ int shared_block_ik[bs*bs];
    __shared__ int shared_block_kk[bs*bs];
    __syncthreads();
    
    int x_shift_kk = k * bs * num_vertices;
    int y_shift = k * bs + threadIdx.y * num_vertices + threadIdx.x;  
    int x_shift_ik = blockIdx.x * bs * num_vertices;
    // int y_shift_ik = k * bs + threadIdx.y * num_vertices + threadIdx.x;  
    
    shared_block_kk[threadIdx.y*bs + threadIdx.x] = dist1[x_shift_kk + y_shift];    // B
    shared_block_ik[threadIdx.y*bs + threadIdx.x] = dist1[x_shift_ik + y_shift];    // C
    
    __syncthreads();
    block_calc(shared_block_ik,shared_block_ik,shared_block_kk,threadIdx.y,threadIdx.x);
    // int tmp;
    // for(int kk=0;kk<bs;kk++){
    //     tmp = shared_block_ik[threadIdx.y*bs + kk] + shared_block_kk[kk*bs + threadIdx.x];  
    //     if(tmp < shared_block_ik[threadIdx.y*bs + threadIdx.x])
    //         shared_block_ik[threadIdx.y*bs + threadIdx.x] = tmp;
    //     __syncthreads();
    // }
    __syncthreads();
    dist1[x_shift_ik + y_shift] = shared_block_ik[threadIdx.y*bs + threadIdx.x];

    __shared__ int shared_block_nkk[bs*bs];  // 
    // int x_shift = k * bs * num_vertices;
    int y_shift_ki = blockIdx.x * bs + threadIdx.y * num_vertices + threadIdx.x;  
    int y_shift_kk = k * bs + threadIdx.y * num_vertices + threadIdx.x;
    // int y_shift_ik = k * bs + threadIdx.y * num_vertices + threadIdx.x;  
    
    shared_block_nkk[threadIdx.y*bs + threadIdx.x] = dist1[x_shift_kk + y_shift_kk];    // A
    shared_block_ik[threadIdx.y*bs + threadIdx.x] = dist1[x_shift_kk + y_shift_ki];     // C
    
    __syncthreads();
    // int tmp;
    block_calc(shared_block_ik,shared_block_nkk,shared_block_ik,threadIdx.y,threadIdx.x);
    // for(int kk=0;kk<bs;kk++){
    //     tmp = shared_block_nkk[threadIdx.y*bs + kk] + shared_block_ik[kk*bs + threadIdx.x];  
    //     if(tmp < shared_block_ik[threadIdx.y*bs + threadIdx.x])
    //         shared_block_ik[threadIdx.y*bs + threadIdx.x] = tmp;
    //     __syncthreads();
    // }
    __syncthreads();
    dist1[x_shift_kk + y_shift_ki] = shared_block_ik[threadIdx.y*bs + threadIdx.x];

}

__global__ void block_fw_p3_kernel(int* dist1, int k, int num_vertices){
    if(blockIdx.x==k&&blockIdx.y==k)
        return;
    __shared__ int shared_block_ij[bs*bs];  // C
    __shared__ int shared_block_ik[bs*bs];  // A
    __shared__ int shared_block_kj[bs*bs];  // B
    __syncthreads();
    int x_shift_i = blockIdx.y * bs * num_vertices;
    int x_shift_k = k * bs * num_vertices;
    int y_shift_j = blockIdx.x * bs + threadIdx.y * num_vertices + threadIdx.x;
    int y_shift_k = k * bs + threadIdx.y * num_vertices + threadIdx.x;
    shared_block_ij[threadIdx.y * bs + threadIdx.x] = dist1[x_shift_i + y_shift_j];
    shared_block_ik[threadIdx.y * bs + threadIdx.x] = dist1[x_shift_i + y_shift_k];
    shared_block_kj[threadIdx.y * bs + threadIdx.x] = dist1[x_shift_k + y_shift_j];

    __syncthreads();
    block_calc(shared_block_ij,shared_block_ik,shared_block_kj,threadIdx.y,threadIdx.x);
    // int tmp;
    // for(int kk=0;kk<bs;kk++){
    //     tmp = shared_block_ik[threadIdx.y*bs + kk] + shared_block_kj[kk*bs + threadIdx.x];  
    //     if(tmp < shared_block_ij[threadIdx.y*bs + threadIdx.x])
    //         shared_block_ij[threadIdx.y*bs + threadIdx.x] = tmp;
    //     __syncthreads();
    // }
    __syncthreads();
    dist1[x_shift_i + y_shift_j] = shared_block_ij[threadIdx.y * bs + threadIdx.x];
}

int* readBinaryFile(char* filename, int& vertices, int& edges, int& pad_vertices) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }
    size_t elements_read;
    elements_read = fread(&vertices, sizeof(int), 1, file);
    elements_read = fread(&edges, sizeof(int), 1, file);
    
    // int* dist = (int**)malloc(vertices * sizeof(int*));
    // if(vertices < bs)
        // bs = vertices;
    if(vertices%bs)
        pad_vertices = vertices + bs - (vertices%bs);
    else
        pad_vertices = vertices;
    int* dist = new int[pad_vertices*pad_vertices];
    for (int i = 0; i < pad_vertices; ++i) {
        // dist[i] = (int*)malloc(vertices * sizeof(int));
        for (int j = 0; j < pad_vertices; ++j) {
            if(i==j&&i<pad_vertices&&j<pad_vertices)
                dist[i*pad_vertices + j] = 0;
            else
                dist[i*pad_vertices + j] = INF;
        }
    }
    int src, dst, weight;
    
    for (int i = 0; i < edges; i++) {
        elements_read = fread(&src, sizeof(int), 1, file);
        elements_read = fread(&dst, sizeof(int), 1, file);
        elements_read = fread(&weight, sizeof(int), 1, file);

        // Assuming vertices are 0-indexed
        dist[src*pad_vertices+dst] = weight;
    }

    fclose(file);
    return dist;
}


void writeBinaryFile(char* FileName, int* dist, int n, int pad_n) {
    FILE* outfile = fopen(FileName, "wb");
 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dist[i*pad_n+j] >= INF) 
                dist[i*pad_n+j] = INF;
        }
        fwrite(dist+i*pad_n, sizeof(int), n, outfile); 
    }

    fclose(outfile);

}

int main(int argc, char* argv[]) {
    char* input_file = argv[1];
    char* output_file = argv[2];
    
    int num_vertices, num_edges, pad_vertices;
    int* dist = readBinaryFile(input_file, num_vertices, num_edges, pad_vertices);
    int* d_dist;
    cudaMalloc((void**)&d_dist, pad_vertices * pad_vertices * sizeof(int));
    cudaMemcpy(d_dist, dist, pad_vertices * pad_vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    int round = pad_vertices/bs; //std::ceil((double)num_vertices / bs);
    dim3 blockSize(bs, bs, 1);
    dim3 gridSize( round, round, 1);
    
    const int blocks = (pad_vertices + bs - 1) / bs;
    // dim3 block_dim(bs, bs, 1);
    dim3 phase4_grid(blocks, blocks, 1);

    for (int k = 0; k < blocks; k++) {
        block_fw_p1_kernel<<<1, blockSize>>>(d_dist, k, pad_vertices);
        block_fw_p2_kernel<<<round, blockSize>>>(d_dist, k, pad_vertices);
        block_fw_p3_kernel<<<gridSize, blockSize>>>(d_dist, k, pad_vertices);
        // floyd_warshall_block_kernel_phase1<<<1, blockSize>>>(pad_vertices, k, d_dist);
        // floyd_warshall_block_kernel_phase2<<<blocks, blockSize>>>(pad_vertices, k, d_dist);
        // floyd_warshall_block_kernel_phase3<<<phase4_grid, blockSize>>>(pad_vertices, k, d_dist);

    }

    cudaMemcpy(dist, d_dist, pad_vertices * pad_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_dist);

    // std::cout<<pad_vertices<<" "<<num_vertices<<std::endl;
    writeBinaryFile(output_file, dist, num_vertices, pad_vertices);
    return 0;
}