#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <limits.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cuda.h>

#define INF 1073741823

// static int bs = 64;
#define BLOCK_DIM 16
#define bs 64
#define BLOCK_SIZE 64
#define HALF_BLOCK_FACTOR 32

using namespace std;

__forceinline__
__device__ void update(int* C, int* A, int* B){
    int tmp = A[0]+B[0]; 
    if(C[0]>tmp)
        C[0] = tmp;
}

__global__ void block_fw_p1_kernel(int* dist1, int k, int num_vertices)
{
    __shared__ int shared_block[bs][bs];  // 添加一列 padding，避免 bank conflicts

    int x_shift = k * bs * num_vertices;

    int tidy = threadIdx.y , tidy_32 = (threadIdx.y + 32);
    int tidy_n = threadIdx.y * num_vertices, tidy_n_32 = (threadIdx.y + 32) * num_vertices;
    int tidx = threadIdx.x, tidx_32 = threadIdx.x + 32;
    int kbs = k * bs; 

    shared_block[tidy][tidx] = dist1[x_shift + kbs + tidy_n + tidx];
    shared_block[tidy_32][tidx] = dist1[x_shift + kbs + tidy_n_32 + tidx];
    shared_block[tidy][tidx_32] = dist1[x_shift + kbs + tidy_n + tidx_32];
    
    shared_block[tidy_32][tidx_32] = dist1[x_shift + kbs + tidy_n_32 + tidx_32];
    __syncthreads();

    for (int kk = 0; kk < bs; kk++) {
        shared_block[tidy][tidx] = min(shared_block[tidy][tidx], shared_block[tidy][kk] + shared_block[kk][tidx]);
        shared_block[tidy][tidx_32] = min(shared_block[tidy][tidx_32], shared_block[tidy][kk] + shared_block[kk][tidx_32]);
        shared_block[tidy_32][tidx] = min(shared_block[tidy_32][tidx], shared_block[tidy_32][kk] + shared_block[kk][tidx]);
        shared_block[tidy_32][tidx_32] = min(shared_block[tidy_32][tidx_32], shared_block[tidy_32][kk] + shared_block[kk][tidx_32]);
        // update(&shared_block[tidy][tidx], &shared_block[tidy][kk], &shared_block[kk][tidx]);
        // update(&shared_block[tidy_32][tidx], &shared_block[tidy_32][kk], &shared_block[kk][tidx]);
        // update(&shared_block[tidy][tidx_32], &shared_block[tidy][kk], &shared_block[kk][tidx_32]);
        
        // update(&shared_block[tidy_32][tidx_32], &shared_block[tidy_32][kk], &shared_block[kk][tidx_32]);

        __syncthreads();
    }
    // __syncthreads();
    dist1[x_shift + kbs + tidy_n + tidx] = shared_block[tidy][tidx];
    dist1[x_shift + kbs + tidy_n_32 + tidx] = shared_block[tidy_32][tidx];
    dist1[x_shift + kbs + tidy_n + tidx_32] = shared_block[tidy][tidx_32];
    
    dist1[x_shift + kbs + tidy_n_32 + tidx_32] = shared_block[tidy_32][tidx_32];
}

__global__ void block_fw_p2_kernel(int* dist1, int k, int num_vertices)
{
    if(blockIdx.x == k)
        return;

    __shared__ int shared_block_ik[bs][bs];
    __shared__ int shared_block_kk[bs][bs];
    __shared__ int shared_block_ki[bs][bs];
    __syncthreads();
    
    int x_shift_kk = k * bs * num_vertices;
    int x_shift_ik = blockIdx.x * bs * num_vertices;

    int tidy = (threadIdx.y), tidy_32 = (threadIdx.y+32);
    int tidy_n = (threadIdx.y)*num_vertices, tidy_n_32 = (threadIdx.y+32)*num_vertices;
    int tidx = threadIdx.x, tidx_32 = threadIdx.x + 32;
    int kbs = k*bs, bbs = blockIdx.x * bs;    

    shared_block_kk[tidy][tidx] = dist1[x_shift_kk + kbs + tidy_n + tidx];
    shared_block_kk[tidy_32][tidx] = dist1[x_shift_kk + kbs + tidy_n_32 + tidx];
    shared_block_kk[tidy][tidx_32] = dist1[x_shift_kk + kbs + tidy_n + tidx_32];
    
    shared_block_kk[tidy_32][tidx_32] = dist1[x_shift_kk + kbs + tidy_n_32 + tidx_32];

    shared_block_ik[tidy][threadIdx.x] = dist1[x_shift_ik + kbs + tidy_n + threadIdx.x];    
    shared_block_ik[tidy_32][threadIdx.x] = dist1[x_shift_ik + kbs + tidy_n_32 + threadIdx.x]; 
    shared_block_ik[tidy][threadIdx.x+32] = dist1[x_shift_ik + kbs + tidy_n + threadIdx.x+32];    
       
    shared_block_ik[tidy_32][threadIdx.x+32] = dist1[x_shift_ik + kbs + tidy_n_32 + threadIdx.x+32];    

    shared_block_ki[tidy][threadIdx.x] = dist1[x_shift_kk + bbs + tidy_n + threadIdx.x];
    shared_block_ki[tidy_32][threadIdx.x] = dist1[x_shift_kk + bbs + tidy_n_32 + threadIdx.x];
    shared_block_ki[tidy][threadIdx.x+32] = dist1[x_shift_kk + bbs + tidy_n + threadIdx.x+32];
    
    shared_block_ki[tidy_32][threadIdx.x+32] = dist1[x_shift_kk + bbs + tidy_n_32 + threadIdx.x+32];    

    __syncthreads();

    
    for(int kk=0;kk<bs;kk++){
        shared_block_ik[tidy][threadIdx.x] = min(shared_block_ik[tidy][threadIdx.x], shared_block_ik[tidy][kk] + shared_block_kk[kk][threadIdx.x]);
        shared_block_ik[tidy_32][threadIdx.x] = min(shared_block_ik[tidy_32][threadIdx.x], shared_block_ik[tidy_32][kk] + shared_block_kk[kk][threadIdx.x]);
        shared_block_ik[tidy][threadIdx.x+32] = min(shared_block_ik[tidy][threadIdx.x+32], shared_block_ik[tidy][kk] + shared_block_kk[kk][tidx_32]);
        shared_block_ik[tidy_32][threadIdx.x+32] = min(shared_block_ik[tidy_32][threadIdx.x+32], shared_block_ik[tidy_32][kk] + shared_block_kk[kk][tidx_32]);
        // update(&shared_block_ik[tidy][threadIdx.x], &shared_block_ik[tidy][kk], &shared_block_kk[kk][threadIdx.x]);
        // update(&shared_block_ik[tidy_32][threadIdx.x], &shared_block_ik[tidy_32][kk], &shared_block_kk[kk][threadIdx.x]);
        // update(&shared_block_ik[tidy][threadIdx.x+32], &shared_block_ik[tidy][kk], &shared_block_kk[kk][tidx_32]);
        // update(&shared_block_ik[tidy_32][threadIdx.x+32], &shared_block_ik[tidy_32][kk], &shared_block_kk[kk][tidx_32]);

        shared_block_ki[tidy][threadIdx.x] = min(shared_block_ki[tidy][threadIdx.x], shared_block_kk[tidy][kk] + shared_block_ki[kk][threadIdx.x]);
        shared_block_ki[tidy_32][threadIdx.x] = min(shared_block_ki[tidy_32][threadIdx.x], shared_block_kk[tidy_32][kk] + shared_block_ki[kk][threadIdx.x]);
        shared_block_ki[tidy][threadIdx.x+32] = min(shared_block_ki[tidy][threadIdx.x+32], shared_block_kk[tidy][kk] + shared_block_ki[kk][tidx_32]);        
        shared_block_ki[tidy_32][threadIdx.x+32] = min(shared_block_ki[tidy_32][threadIdx.x+32], shared_block_kk[tidy_32][kk] + shared_block_ki[kk][tidx_32]);
        
        // update(&shared_block_ki[tidy][threadIdx.x], &shared_block_kk[tidy][kk], &shared_block_ki[kk][threadIdx.x]);
        // update(&shared_block_ki[tidy_32][threadIdx.x], &shared_block_kk[tidy_32][kk], &shared_block_ki[kk][threadIdx.x]);
        // update(&shared_block_ki[tidy][threadIdx.x+32], &shared_block_kk[tidy][kk], &shared_block_ki[kk][tidx_32]);        
        // update(&shared_block_ki[tidy_32][threadIdx.x+32], &shared_block_kk[tidy_32][kk], &shared_block_ki[kk][tidx_32]);
    }

    dist1[x_shift_ik + kbs + tidy_n + threadIdx.x] = shared_block_ik[tidy][threadIdx.x];    
    dist1[x_shift_ik + kbs + tidy_n_32 + threadIdx.x] = shared_block_ik[tidy_32][threadIdx.x];    
    dist1[x_shift_ik + kbs + tidy_n + threadIdx.x+32] = shared_block_ik[tidy][threadIdx.x+32];    
    
    dist1[x_shift_ik + kbs + tidy_n_32 + threadIdx.x+32] = shared_block_ik[tidy_32][threadIdx.x+32];    

    dist1[x_shift_kk + bbs + tidy_n + threadIdx.x] = shared_block_ki[tidy][threadIdx.x];
    dist1[x_shift_kk + bbs + tidy_n_32 + threadIdx.x] = shared_block_ki[tidy_32][threadIdx.x];
    dist1[x_shift_kk + bbs + tidy_n + threadIdx.x+32] = shared_block_ki[tidy][threadIdx.x+32];
    
    dist1[x_shift_kk + bbs + tidy_n_32 + threadIdx.x+32] = shared_block_ki[tidy_32][threadIdx.x+32]; 

}

__global__ void block_fw_p3_kernel(int* dist1, int k, int num_vertices){
    if(blockIdx.x==k&&blockIdx.y==k)
        return;

    __shared__ int shared_block_ij[bs][bs];  // C
    __shared__ int shared_block_ik[bs][bs];  // A
    __shared__ int shared_block_kj[bs][bs];  // B
    __syncthreads();

    int x_shift_i = blockIdx.y * bs * num_vertices;
    int x_shift_k = k * bs * num_vertices;

    int tidy = (threadIdx.y), tidy_32 = (threadIdx.y+32);
    int tidy_n = (threadIdx.y)*num_vertices, tidy_n_32 = (threadIdx.y+32)*num_vertices;
    int tidx = threadIdx.x, tidx_32 = threadIdx.x + 32;
    int kbs = k*bs, bbs = blockIdx.x * bs; 

    shared_block_ij[tidy][threadIdx.x] = dist1[x_shift_i + bbs + tidy_n + threadIdx.x];
    shared_block_ij[tidy_32][threadIdx.x] = dist1[x_shift_i + bbs + tidy_n_32 + threadIdx.x];
    shared_block_ij[tidy][tidx_32] = dist1[x_shift_i + bbs + tidy_n + tidx_32];
    
    shared_block_ij[tidy_32][tidx_32] = dist1[x_shift_i + bbs + tidy_n_32 + tidx_32];

    shared_block_ik[tidy][threadIdx.x] = dist1[x_shift_i + kbs + tidy_n + threadIdx.x];
    shared_block_ik[tidy_32][threadIdx.x] = dist1[x_shift_i + kbs + tidy_n_32 + threadIdx.x];
    shared_block_ik[tidy][tidx_32] = dist1[x_shift_i + kbs + tidy_n + tidx_32];
    
    shared_block_ik[tidy_32][tidx_32] = dist1[x_shift_i + kbs + tidy_n_32 + tidx_32];

    shared_block_kj[tidy][threadIdx.x] = dist1[x_shift_k + bbs + tidy_n + threadIdx.x];
    shared_block_kj[tidy_32][threadIdx.x] = dist1[x_shift_k + bbs + tidy_n_32 + threadIdx.x];
    shared_block_kj[tidy][tidx_32] = dist1[x_shift_k + bbs + tidy_n + tidx_32];
    
    shared_block_kj[tidy_32][tidx_32] = dist1[x_shift_k + bbs + tidy_n_32 + tidx_32];


    __syncthreads();

    for(int kk=0;kk<bs;kk++){
        shared_block_ij[tidy][threadIdx.x] = min(shared_block_ij[tidy][threadIdx.x], shared_block_ik[tidy][kk] + shared_block_kj[kk][threadIdx.x]);
        shared_block_ij[tidy_32][threadIdx.x] = min(shared_block_ij[tidy_32][threadIdx.x], shared_block_ik[tidy_32][kk] + shared_block_kj[kk][threadIdx.x]);
        shared_block_ij[tidy][tidx_32] = min(shared_block_ij[tidy][tidx_32], shared_block_ik[tidy][kk] + shared_block_kj[kk][tidx_32]);
        
        shared_block_ij[tidy_32][tidx_32] = min(shared_block_ij[tidy_32][tidx_32], shared_block_ik[tidy_32][kk] + shared_block_kj[kk][tidx_32]);
    }

    dist1[x_shift_i + bbs + tidy_n + threadIdx.x] = shared_block_ij[tidy][threadIdx.x];
    dist1[x_shift_i + bbs + tidy_n_32 + threadIdx.x] = shared_block_ij[tidy_32][threadIdx.x];
    dist1[x_shift_i + bbs + tidy_n + tidx_32] = shared_block_ij[tidy][tidx_32];
    
    dist1[x_shift_i + bbs + tidy_n_32 + tidx_32] = shared_block_ij[tidy_32][tidx_32];


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
            // if (dist[i*pad_n+j] >= INF) 
                // dist[i*pad_n+j] = INF;
            dist[i*pad_n+j] = (dist[i*pad_n+j] >= INF) ? INF : dist[i*pad_n+j];

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
    cudaHostRegister(dist, pad_vertices*pad_vertices*sizeof(int), cudaHostRegisterDefault);
    cudaMalloc((void**)&d_dist, pad_vertices * pad_vertices * sizeof(int));
    cudaMemcpy(d_dist, dist, pad_vertices * pad_vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    int round = (pad_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockSize(32, 32, 1);
    dim3 gridSize( round, round, 1);

    for (int k = 0; k < round; k++) {
        // std::cout<<k<<std::endl;
        block_fw_p1_kernel<<<1, blockSize>>>(d_dist, k, pad_vertices);
        // Phase1<<<1, blockSize>>>(d_dist,k,pad_vertices);
        // Phase2<<<round, blockSize>>>(d_dist,k,pad_vertices);
        // Phase3<<<gridSize, blockSize>>>(d_dist,k,pad_vertices);
        block_fw_p2_kernel<<<round, blockSize>>>(d_dist, k, pad_vertices);
        block_fw_p3_kernel<<<gridSize, blockSize>>>(d_dist, k, pad_vertices);


    }

    cudaMemcpy(dist, d_dist, pad_vertices * pad_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_dist);

    // std::cout<<pad_vertices<<" "<<num_vertices<<std::endl;
    writeBinaryFile(output_file, dist, num_vertices, pad_vertices);
    return 0;
}