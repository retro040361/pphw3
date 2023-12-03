#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <limits.h>
#include <algorithm>
#include <cmath>

#define INF 1073741823


int** readBinaryFile(char* filename, int& vertices, int& edges) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }
    
    fread(&vertices, sizeof(int), 1, file);
    fread(&edges, sizeof(int), 1, file);
    
    int** dist = (int**)malloc(vertices * sizeof(int*));

    for (int i = 0; i < vertices; ++i) {
        dist[i] = (int*)malloc(vertices * sizeof(int));
        for (int j = 0; j < vertices; ++j) {
            if(i==j)
                dist[i][j] = 0;
            else
                dist[i][j] = INF;
        }
    }
    int src, dst, weight;
    for (int i = 0; i < edges; i++) {
        fread(&src, sizeof(int), 1, file);
        fread(&dst, sizeof(int), 1, file);
        fread(&weight, sizeof(int), 1, file);

        // Assuming vertices are 0-indexed
        dist[src][dst] = weight;
    }

    fclose(file);
    return dist;
}

// std::vector<std::vector<int>> readBinaryFile(const char* filename, int& vertices, int& edges) {
    
//     FILE* file = fopen(filename, "rb");
//     if (file == NULL) {
//         perror("Error opening file");
//         exit(1);
//     }

//     fread(&vertices, sizeof(int), 1, file);
//     fread(&edges, sizeof(int), 1, file);
//     std::vector<std::vector<int>> dist(num_vertices,std::vector<int>(num_vertices,INF));
//     // dist.resize(vertices, std::vector<int>(vertices, 0));

//     for (int i = 0; i < edges; i++) {
//         int src, dst, weight;
//         fread(&src, sizeof(int), 1, file);
//         fread(&dst, sizeof(int), 1, file);
//         fread(&weight, sizeof(int), 1, file);

//         // Assuming vertices are 0-indexed
//         dist[src][dst] = weight;
//     }

//     fclose(file);
//     return dist;
// }

void writeBinaryFile(char* FileName, int** dist, int n) {
    FILE* outfile = fopen(FileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dist[i][j] >= INF) 
                dist[i][j] = INF;
        }
        fwrite(dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

inline void block_fw(int* dist1, int* dist2, int* dist3 ,int block_h, int block_w, int num_vertices){
    int kn, in, tmp;
    int blockSize = std::min(block_h, block_w);
    for(int k=0;k<blockSize;k++){
        kn = k*num_vertices;
        for(int i=0;i<block_w;i++){
            in = i*num_vertices;
            for(int j=0;j<block_h;j++){
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
    // std::vector<std::vector<int>> dist = readBinaryFile(input_file, num_vertices, num_edges);
    int** dist = readBinaryFile(input_file, num_vertices, num_edges);

    // // int num_thread = omp_get_thread_num();
    // int bs = 32; // blocksize // 512
    // int round = std::ceil(num_vertices / bs);
    // int block_h = bs;
    // int block_w = bs;
    // int k_bs;
    // for(int k = 0;k<round;k++){        
    //     k_bs = k*bs;
    //     // phase 1

    //     if((k+1)*bs>=num_vertices){
    //         block_fw(&dist[k_bs][k_bs],&dist[k_bs][k_bs],&dist[k_bs][k_bs],num_vertices-(k_bs), num_vertices-(k_bs), num_vertices);
    //     }
    //     else{
    //         block_fw(&dist[k_bs][k_bs],&dist[k_bs][k_bs],&dist[k_bs][k_bs],bs, bs, num_vertices);
    //     }

    //     // phase 2 : row
    //     for(int j=0;j<round;j++){
    //         if(j!=k){
    //             if((j+1)*bs>num_vertices)
    //                 block_fw(&dist[k_bs][j*bs],&dist[k_bs][k_bs],&dist[k_bs][j*bs],bs,num_vertices-j*bs, num_vertices);
    //             else
    //                 block_fw(&dist[k_bs][j*bs],&dist[k_bs][k_bs],&dist[k_bs][j*bs],bs,bs,num_vertices);
    //         }
    //     }

    //     for(int i=0;i<round;i++){
    //         // phase 2: col
    //         if(i!=k){
    //             if((i+1)*bs>num_vertices)
    //                 block_fw(&dist[i*bs][k_bs],&dist[i*bs][k_bs],&dist[k_bs][k_bs],num_vertices-i*bs,bs, num_vertices);
    //             else
    //                 block_fw(&dist[i*bs][k_bs],&dist[i*bs][k_bs],&dist[k_bs][k_bs],bs,bs,num_vertices);
    //         }
    //         // phase 3
    //         for(int j=0;j<round;j++){
    //             if(j!=k){
    //                 block_h = ((i+1)*bs>num_vertices)? num_vertices-i*bs:bs;
    //                 block_w = ((j+1)*bs>num_vertices)? num_vertices-j*bs:bs;
    //                 block_fw(&dist[i*bs][j*bs],&dist[i*bs][k_bs],&dist[k_bs][j*bs],block_h,block_w, num_vertices);
    //             }
    //         }
    //     }
    // }

    writeBinaryFile(output_file, dist, num_vertices);

    return 0;
}
