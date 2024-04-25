#include <algorithm>
#include <cmath>
#include <emmintrin.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <nmmintrin.h>
#include <numeric>
#include <pmmintrin.h>
#include <random>
#include <smmintrin.h>
#include <sys/time.h>
#include <tmmintrin.h>
#include <vector>
#include <xmmintrin.h>
using namespace std;
enum DisType {
  EucDis, // 欧氏距离
  Cosine,
  ManDis, // 曼哈顿距离
};
unsigned dim, num;
void load_ivecs_data(const char *filename, vector<vector<float>> &results,
                     unsigned &num, unsigned &dim) {
  ifstream in(filename, ios::binary);
  if (!in.is_open()) {
    cout << "open file error" << endl;
    exit(-1);
  }
  in.read((char *)&dim, 4);

  in.seekg(0, ios::end);
  ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  results.resize(num);
  for (unsigned i = 0; i < num; i++)
    results[i].resize(dim);
  in.seekg(0, ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, ios::cur);
    in.read((char *)results[i].data(), dim * 4);
  }
  in.close();
}
void split_data(vector<vector<float>> &ori_data,
                vector<vector<vector<float>>> &split_data,
                unsigned sub_data_size, unsigned size) {
  unsigned sub_dim = dim / sub_data_size;
  for (unsigned i = 0; i < size; i++) {
    for (unsigned j = 0; j < dim; j++) {
      unsigned idx = j / sub_dim;
      split_data[idx][i].push_back(ori_data[i][j]);
    }
  }
}

float dist(const std::vector<float> &v1, const std::vector<float> &v2, DisType type) {
    unsigned int size = v1.size();
    unsigned int numVectors = size / 8; // 每次处理8个float，计算向量个数
    __m256 sumVec = _mm256_setzero_ps(); // 使用256位的寄存器

    switch (type) {
    case EucDis:
        for (unsigned int i = 0; i < numVectors; ++i) {
            // 加载两个向量
            __m256 v1Vec = _mm256_loadu_ps(&v1[i * 8]);
            __m256 v2Vec = _mm256_loadu_ps(&v2[i * 8]);
            // 计算差值
            __m256 diff = _mm256_sub_ps(v1Vec, v2Vec);
            // 计算差值的平方
            __m256 squared = _mm256_mul_ps(diff, diff);
            // 累加平方差值到sumVec
            sumVec = _mm256_add_ps(sumVec, squared);
        }
        break;
    default:
        break;
    }

    // 水平加和
    __m256 sumHorizontal = _mm256_hadd_ps(sumVec, sumVec);
    sumHorizontal = _mm256_hadd_ps(sumHorizontal, sumHorizontal);
    // 将结果存储到临时数组中
    float tempSum[8];
    _mm256_storeu_ps(tempSum, sumHorizontal);
    // 对临时数组中的值求和
    float sum = tempSum[0] + tempSum[4];

    sum = sqrt(sum); // 对结果求平方根
    return sum;
}
vector<vector<float>> initCentroids(const vector<vector<float>> &data,
                                    unsigned k, unsigned seed) {
  std::mt19937 gen(seed); // 使用确定性种子生成器
  std::uniform_int_distribution<> dis(0, data.size() - 1);

  std::vector<std::vector<float>> centroids(k);
  for (unsigned i = 0; i < k; i++) {
    int idx = dis(gen);
    centroids[i] = data[idx];
  }
  return centroids;
}

vector<int> kmeans(const vector<vector<float>> &data, unsigned k,
                   vector<vector<float>> &centroid) {
  auto centroids = initCentroids(data, k, 42);
  vector<int> clusters(data.size());
  while (true) {
    for (unsigned i = 0; i < data.size(); i++) {
      float minDist = dist(data[i], centroids[0], EucDis);
      int nearestCentroid = 0;
      for (int j = 1; j < k; j++) {
        float distance = dist(data[i], centroids[j], EucDis);
        if (distance < minDist) {
          minDist = distance;
          nearestCentroid = j;
        }
      }
      clusters[i] = nearestCentroid;
    }

    auto prev = clusters;

    for (int i = 0; i < k; i++) {
      centroids[i].clear();
      centroids[i].resize(data[0].size(), 0.0);
    }

    for (int i = 0; i < clusters.size(); i++) {
      for (int j = 0; j < data[i].size(); j++) {
        centroids[clusters[i]][j] += data[i][j];
      }
    }

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < centroids[i].size(); j++) {
        centroids[i][j] /= clusters.size();
      }
    }


    bool converged = true;
    for (int i = 0; i < clusters.size(); i++) {
      if (clusters[i] != prev[i]) {
        converged = false;
        break;
      }
    }
    if (converged) {
      break;
    }
  }
  return clusters;
}

int main() {
  //   char path[1024] = "data/sift_base.fvecs";
  //   vector<vector<float>> ori_data;

  //   load_ivecs_data(path, ori_data, num, dim);
  //   int sub_data_size = 4;

  //   int rows = 50000;

  //   vector<vector<vector<float>>> sub_data(sub_data_size,
  //                                          vector<vector<float>>(rows));
  //   split_data(ori_data, sub_data, 4, rows);

  //   cout << "result_num：" << num << endl << "result dimension：" << dim <<
  //   endl; struct timeval t1, t2; double timeuse; gettimeofday(&t1, NULL);
  //   vector<vector<int>> clusters(sub_data_size);
  //   vector<vector<float>> centroids(dim / sub_data_size);
  //   for (int i = 0; i < sub_data_size; i++) {
  //     clusters[i] = kmeans(sub_data[i], 256, centroids);
  //   }
  //   gettimeofday(&t2, NULL);
  //   timeuse =
  //       (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) /
  //       1000000.0;
  //   cout << timeuse << " seconds \n";

  //   for (int i = 0; i < 5; i++) {
  //     for (int k = 0; k < 4; k++) {
  //       cout << clusters[k][i] << " ";
  //     }
  //     cout << endl;
  //   }

  vector<float> v2{1, 2, 3, 4, 4, 4, 5, 6};
  vector<float> v1{1, 2, 3, 3, 5, 7, 8, 6};
  cout << dist(v1, v2, EucDis) << endl;

  return 0;
}

// avx代码