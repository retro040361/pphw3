#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <limits.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#define INF 1073741823

// const int bs = 512;
#define bs 512
// const int num_threads = 16; // 調整為您需要的數量

__global__ void block_fw_kernel(int* dist1, int* dist2, int* dist3, int block_h, int block_w, int num_vertices) {
    int kn, in, tmp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < block_h && j < block_w) {
        kn = threadIdx.z * num_vertices;
        in = i * num_vertices;
        tmp = dist2[in + threadIdx.z] + dist3[kn + j];

        if (dist1[in + j] > tmp)
            dist1[in + j] = tmp;
    }
}

__global__ void block_fw_p3_kernel(int* dist1, int* dist2, int* dist3, int block_h, int block_w, int num_vertices) {
    int kn, in, tmp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = threadIdx.z;

    if (i < block_h && j < block_w && k < num_vertices) {
        kn = k * num_vertices;
        in = i * num_vertices;
        tmp = dist2[in + k] + dist3[kn + j];

        if (dist1[in + j] > tmp)
            dist1[in + j] = tmp;
    }
}

__global__ void block_fw_p1_kernel(int* dist1, int k, int num_vertices)
{
    __shared__ int shared_block[bs*bs];
    __syncthreads();
    
    int x_shift = k*bs;
    int y_shift = k*bs + threadIdx.y * num_vertices + threadIdx.x;  //
    shared_block[threadIdx.y*bs + threadIdx.x] = dist1[x_shift*num_vertices + y_shift];
    __syncthreads();
    int tmp;
    for(int k=0;k<bs;k++){
        tmp = shared_block[threadIdx.y*num_vertices + k] + shared_block[k*num_vertices + threadIdx.x];  
        if(tmp < shared_block[threadIdx.y*num_vertices + threadIdx.x])
            shared_block[threadIdx.y*num_vertices + threadIdx.x] = tmp;
        __syncthreads();
    }
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
    
    shared_block_kk[threadIdx.y*bs + threadIdx.x] = dist1[x_shift_kk + y_shift];
    shared_block_ik[threadIdx.y*bs + threadIdx.x] = dist1[x_shift_ik + y_shift];
    
    __syncthreads();
    int tmp;
    for(int k=0;k<bs;k++){
        tmp = shared_block_ik[threadIdx.y*num_vertices + k] + shared_block_kk[k*num_vertices + threadIdx.x];  
        if(tmp < shared_block_ik[threadIdx.y*num_vertices + threadIdx.x])
            shared_block_ik[threadIdx.y*num_vertices + threadIdx.x] = tmp;
        __syncthreads();
    }
    __syncthreads();
    dist1[x_shift_ik + y_shift] = shared_block_ik[threadIdx.y*bs + threadIdx.x];

    __shared__ int shared_block_ki[bs*bs];
    // int x_shift = k * bs * num_vertices;
    int y_shift_ki = blockIdx.x * bs + threadIdx.y * num_vertices + threadIdx.x;  
    int y_shift_kk = k * bs + threadIdx.y * num_vertices + threadIdx.x;
    // int y_shift_ik = k * bs + threadIdx.y * num_vertices + threadIdx.x;  
    
    shared_block_kk[threadIdx.y*bs + threadIdx.x] = dist1[x_shift_kk + y_shift_kk];
    shared_block_ki[threadIdx.y*bs + threadIdx.x] = dist1[x_shift_kk + y_shift_ki];
    
    __syncthreads();
    // int tmp;
    for(int k=0;k<bs;k++){
        tmp = shared_block_kk[threadIdx.y*num_vertices + k] + shared_block_ki[k*num_vertices + threadIdx.x];  
        if(tmp < shared_block_ki[threadIdx.y*num_vertices + threadIdx.x])
            shared_block_ki[threadIdx.y*num_vertices + threadIdx.x] = tmp;
        __syncthreads();
    }
    __syncthreads();
    dist1[x_shift_kk + y_shift_ki] = shared_block_ki[threadIdx.y*bs + threadIdx.x];

}

__global__ void block_fw_p3_kernel(int* dist1, int k, int num_vertices){
    if(i==k||j==k)
        return;
    __shared__ shared_block_ij[bs*bs];
    __shared__ shared_block_ik[bs*bs];
    __shared__ shared_block_kj[bs*bs];
    __syncthreads();
    int x_shift_i = blockIdx.y * bs * num_vertices;
    int x_shift_k = k * bs * num_vertices;
    int y_shift_j = blockIdx.x * bs + threadIdx.y * num_vertices + threadIdx.x;
    int y_shift_k = k * bs + threadIdx.y * num_vertices + threadIdx.x;
    shared_block_ij[threadIdx.y * bs + threadIdx.x] = dist1[x_shift_i + y_shift_j];
    shared_block_ik[threadIdx.y * bs + threadIdx.x] = dist1[x_shift_i + y_shift_k];
    shared_block_kj[threadIdx.y * bs + threadIdx.x] = dist1[x_shift_k + y_shift_j];

    __syncthreads();
    int tmp;
    for(int k=0;k<bs;k++){
        tmp = shared_block_ik[threadIdx.y*num_vertices + k] + shared_block_kj[k*num_vertices + threadIdx.x];  
        if(tmp < shared_block_ij[threadIdx.y*num_vertices + threadIdx.x])
            shared_block_ij[threadIdx.y*num_vertices + threadIdx.x] = tmp;
        __syncthreads();
    }
    __syncthreads();
    dist1[x_shift_i + y_shift_j] = shared_block_ij[threadIdx.y * bs + threadIdx.x];
}

int* readBinaryFile(char* filename, int& vertices, int& edges, int& pad_vertices) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }
    
    fread(&vertices, sizeof(int), 1, file);
    fread(&edges, sizeof(int), 1, file);
    
    // int* dist = (int**)malloc(vertices * sizeof(int*));
    if(vertices%bs)
        pad_vertices = vertices + bs + vertices%bs;
    else
        pad_vertices = vertices;
    int* dist = new int[pad_vertices*pad_vertices];
    for (int i = 0; i < pad_vertices; ++i) {
        // dist[i] = (int*)malloc(vertices * sizeof(int));
        for (int j = 0; j < pad_vertices; ++j) {
            if(i==j&&i<vertices&&j<vertices)
                dist[i*vertices + j] = 0;
            else
                dist[i*vertices + j] = INF;
        }
    }
    int src, dst, weight;
    for (int i = 0; i < edges; i++) {
        fread(&src, sizeof(int), 1, file);
        fread(&dst, sizeof(int), 1, file);
        fread(&weight, sizeof(int), 1, file);

        // Assuming vertices are 0-indexed
        dist[src*vertices+dst] = weight;
    }

    fclose(file);
    return dist;
}


void writeBinaryFile(char* FileName, int* dist, int n) {
    FILE* outfile = fopen(FileName, "wb");
    // #pragma omp parallel for num_threads(num_threads)  schedule(dynamic, 1) 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dist[i*n+j] >= INF) 
                dist[i*n+j] = INF;
        }
        fwrite(dist[i], sizeof(int), n, outfile); 
    }
    // std::cout<<"hi"<<std::endl;
    // fwrite(dist, sizeof(int), n*n, outfile);
    // std::cout<<"write done"<<std::endl;
    fclose(outfile);
    // std::cout<<"fclose"<<std::endl;
}

int main(int argc, char* argv[]) {
    char* input_file = argv[1];
    char* output_file = argv[2];
    
    int num_vertices, num_edges, pad_vertices;
    int* dist = readBinaryFile(input_file, num_vertices, num_edges, pad_vertices);
    int* d_dist;
    cudaMalloc((void**)&d_dist, pad_vertices * pad_vertices * sizeof(int));
    cudaMemcpy(d_dist, dist, pad_vertices * pad_vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    int round = std::ceil((double)num_vertices / bs);
    dim3 blockSize(bs, bs, 1);
    dim3 gridSize( round, round, 1);


    for (int k = 0; k < round; k++) {
        block_fw_p1_kernel<<1, blockSize>>(d_dist, k, pad_vertices);
        block_fw_p2_kernel<<round, blockSize>>(d_dist, k, pad_vertices);
        block_fw_p3_kernel<<gridSize, blockSize>>(d_dist, k, pad_vertices);
    }

    cudaMemcpy(dist, d_dist, pad_vertices * pad_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_dist);


    writeBinaryFile(output_file, dist, num_vertices);
    return 0;
}
