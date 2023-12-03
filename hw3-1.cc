#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <limits.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#define INF 1073741823

int bs = 512;
int num_threads;

int* readBinaryFile(char* filename, int& vertices, int& edges) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }
    
    fread(&vertices, sizeof(int), 1, file);
    fread(&edges, sizeof(int), 1, file);
    
    // int* dist = (int**)malloc(vertices * sizeof(int*));
    int* dist = new int[vertices*vertices];
    for (int i = 0; i < vertices; ++i) {
        // dist[i] = (int*)malloc(vertices * sizeof(int));
        for (int j = 0; j < vertices; ++j) {
            if(i==j)
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
    #pragma omp parallel for num_threads(num_threads)  schedule(dynamic, 1) 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dist[i*n+j] >= INF) 
                dist[i*n+j] = INF;
        }
        
    }
    // std::cout<<"hi"<<std::endl;
    fwrite(dist, sizeof(int), n*n, outfile);
    // std::cout<<"write done"<<std::endl;
    fclose(outfile);
    // std::cout<<"fclose"<<std::endl;
}

inline void block_fw(int* dist1, int* dist2, int* dist3 ,int block_h, int block_w, int num_vertices){
    int kn, in, tmp;
    int blockSize = std::min(block_h, block_w);
    for(int k=0;k<blockSize;k++){
        kn = k*num_vertices;
        for(int i=0;i<block_h;i++){
            in = i*num_vertices;
            for(int j=0;j<block_w;j++){
                tmp = dist2[in+k]+dist3[kn+j];
                // ik+kj
                if(dist1[in+j]>tmp)
                    dist1[in+j] = tmp;
            }
        }
    }
}

inline void block_fw_p3(int* dist1, int* dist2, int* dist3 , std::pair<int,int>bhw1, std::pair<int,int>bhw2, std::pair<int,int>bhw3, int num_vertices){
    // first: block_h, second: block_w
    int kn, in, tmp;
    int blockSize = std::min(bhw3.first, bhw2.second);
    for(int k=0;k<blockSize;k++){
        if(k>bhw2.second||k>bhw3.first)
            break;
        kn = k*num_vertices;
        
        for(int i=0;i<bhw1.first;i++){
            if(i>bhw2.first)
                break;
            in = i*num_vertices;
            for(int j=0;j<bhw1.second;j++){
                if(j>bhw3.second)
                    break;
                tmp = dist2[in+k]+dist3[kn+j];
                // ik+kj
                if(dist1[in+j]>tmp)
                    dist1[in+j] = tmp;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    
    char* input_file = argv[1];
    char* output_file = argv[2];
    
    int num_vertices, num_edges;

    int* dist = readBinaryFile(input_file, num_vertices, num_edges);

    // int bs = 20; // blocksize // 512
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = omp_get_max_threads();
    if(bs>num_vertices)
        bs = num_vertices;
    int round = std::ceil((double)num_vertices / bs);
    int block_h = bs;
    int block_w = bs;
    int block_k = bs;
    int k_bs,j_bs,i_bs;
    int k_bsn;
    for(int k = 0;k<round;k++){        

        k_bs = k*bs;
        k_bsn = k_bs * num_vertices;
        block_k = (k_bs + bs) > num_vertices ? num_vertices - k_bs:bs;
        // phase 1

        if((k+1)*bs>num_vertices){
            block_fw(&dist[k_bsn + k_bs],&dist[k_bsn+k_bs],&dist[k_bsn+k_bs],num_vertices-(k_bs), num_vertices-(k_bs), num_vertices);
        }
        else{
            block_fw(&dist[k_bsn + k_bs],&dist[k_bsn+k_bs],&dist[k_bsn+k_bs],bs, bs, num_vertices);
        }
        #pragma omp parallel for num_threads(num_threads)  schedule(dynamic, 1) 
        for(int j=0;j<round;j++){
            if(j!=k){
                block_w = (j+1)*bs>num_vertices ? num_vertices - j*bs:bs;
                block_fw_p3(&dist[k_bsn+j*bs],&dist[k_bsn+k_bs],&dist[k_bsn+j*bs], std::pair(block_k,block_w), std::pair(block_k, block_k), std::pair(block_k, block_w),num_vertices);
            }
        }

        for(int i=0;i<round;i++){
            if(i!=k){

                block_h = (i+1)*bs>num_vertices? num_vertices-i*bs:bs;

                block_fw_p3(&dist[i*bs*num_vertices+k_bs],&dist[i*bs*num_vertices+k_bs],&dist[k_bs*num_vertices+k_bs], std::pair(block_h,block_k), std::pair(block_h, block_k), std::pair(block_k, block_k),num_vertices);
                // #pragma omp parallel for schedule(dynamic)
                #pragma omp parallel for num_threads(num_threads)  schedule(dynamic, 1)
                for(int j=0;j<round;j++){
                    if(j!=k){
                        block_w = (j+1)*bs>num_vertices? num_vertices-j*bs:bs; 
                        block_fw_p3(&dist[i*bs*num_vertices+j*bs],&dist[i*bs*num_vertices+k_bs],&dist[k_bs*num_vertices+j*bs], std::pair(block_h,block_w), std::pair(block_h, block_k), std::pair(block_k, block_w),num_vertices);
                    }
                    
                }
            }
        }
    }

    
    writeBinaryFile(output_file, dist, num_vertices);
    
    delete []dist;
    return 0;
}
