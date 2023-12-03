#include <iostream>
#include <fstream>

#define INF 999999

void readBinaryFile(const char* fileName, int n) {
    std::ifstream infile(fileName, std::ios::binary);
    // int idx=0;
    if (!infile) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        exit(1);
    }

    int value;
    int idx=0;
    while (infile.read(reinterpret_cast<char*>(&value), sizeof(int))) {
        std::cout << value << " ";
        if((++idx)%n==0)
            std::cout<<std::endl;
    }

    infile.close();
}

int main(int argc, char* argv[]) {
    const char* fileName = argv[1]; // 替換為你的文件名

    int** dist;
    int n = atoi(argv[2]);
    std::cout<<n<<std::endl;
    readBinaryFile(fileName, n);

    // 現在 dist 包含了從二進制文件中讀取的數據
    // for(int i =0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         std::cout<<dist[i][j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    // // 使用完畢後，記得釋放動態分配的記憶體
    // for (int i = 0; i < n; ++i) {
    //     delete[] dist[i];
    // }
    // delete[] dist;

    return 0;
}
