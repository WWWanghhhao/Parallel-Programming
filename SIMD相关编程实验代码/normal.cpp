#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <sys/time.h>
#include <vector>
using namespace std;
enum DisType {
  EucDis, // 欧氏距离
  Cosine,
  ManDis, // 曼哈顿距离

};
int dim, num;
void load_ivecs_data(const char *filename, vector<vector<float>> &results,
                     int &num, int &dim) {
  ifstream in(filename, ios::binary);
  if (!in.is_open()) {
    cout << "open file error" << endl;
    exit(-1);
  }
  in.read((char *)&dim, 4);

  in.seekg(0, ios::end);
  ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (int)(fsize / (dim + 1) / 4);
  results.resize(num);
  for (int i = 0; i < num; i++)
    results[i].resize(dim);
  in.seekg(0, ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, ios::cur);
    in.read((char *)results[i].data(), dim * 4);
  }
  in.close();
}
void split_data(vector<vector<float>> &ori_data,
                vector<vector<vector<float>>> &split_data, int sub_data_size,
                int size) {
  int sub_dim = dim / sub_data_size;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < dim; j++) {
      int idx = j / sub_dim;
      split_data[idx][i].push_back(ori_data[i][j]);
    }
  }
}
float dist(vector<float> v1, vector<float> v2, DisType type) {

  float sum = 0.0;
  float m1 = 0, m2 = 0;
  switch (type) {
  case EucDis:
    for (int i = 0; i < v1.size(); i++) {
      sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    sum = sqrt(sum);
    break;
  case Cosine:
    for (int i = 0; i < v1.size(); i++) {
      sum += v1[i] * v2[i];
    }
    for (int i = 0; i < v1.size(); i++) {
      m1 += v1[i] * v1[i];
      m2 += v2[i] * v2[i];
    }
    m1 = sqrt(m1);
    m2 = sqrt(m2);
    sum = sum / (m1 * m2);
    break;
  case ManDis:
    for (int i = 0; i < v1.size(); i++) {
      sum += abs(v1[i] - v2[i]);
    }
    break;
  }
  return sum;
}

vector<vector<float>> initCentroids(const vector<vector<float>> &data, int k) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, data.size() - 1);

  std::vector<std::vector<float>> centroids(k);
  for (int i = 0; i < k; i++) {
    int idx = dis(gen);
    centroids[i] = data[idx];
  }
  return centroids;
}

vector<int> kmeans(const vector<vector<float>> &data, int k,
                   vector<vector<float>> &centroid) {
  auto centroids = initCentroids(data, k);
  vector<int> clusters(data.size());
  while (true) {
    for (int i = 0; i < data.size(); i++) {
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
      centroid = centroids;
      break;
    }
  }
  return clusters;
}

int main() {
  char path[1024] = "data/sift_base.fvecs";
  vector<vector<float>> ori_data;

  load_ivecs_data(path, ori_data, num, dim);
  int sub_data_size = 4;

  int rows = 50000;

  vector<vector<vector<float>>> sub_data(sub_data_size,
                                         vector<vector<float>>(rows));
  split_data(ori_data, sub_data, 4, rows);

  cout << "result_num：" << num << endl << "result dimension：" << dim << endl;
  struct timeval t1, t2;
  double timeuse;
  gettimeofday(&t1, NULL);
  vector<vector<int>> clusters(sub_data_size);
  vector<vector<vector<float>>> centroids(sub_data_size);
  for (int i = 0; i < sub_data_size; i++) {
    clusters[i] = kmeans(sub_data[i], 256, centroids[i]);
  }
  gettimeofday(&t2, NULL);
  timeuse =
      (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
  cout << timeuse << " seconds \n";


  return 0;
}