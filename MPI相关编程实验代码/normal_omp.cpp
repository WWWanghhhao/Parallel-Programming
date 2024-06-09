#include "/usr/local/openmpi/include/mpi.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <emmintrin.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <nmmintrin.h>
#include <numeric>
#include <pmmintrin.h>
#include <random>
#include <smmintrin.h>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <tmmintrin.h>
#include <vector>
#include <xmmintrin.h>
#include <omp.h>
using namespace std;
enum DisType {
  EucDis, // 欧氏距离
  Cosine,
  ManDis, // 曼哈顿距离
};
int dim, num;
int size = 100;
int sub_data_size = 4;
char path[1024] = "../data/sift_query.fvecs";
vector<vector<float>> ori_data;
vector<vector<vector<float>>> centroids(sub_data_size);
vector<vector<int>> clusters(sub_data_size);

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
float dist(const std::vector<float> &v1, const std::vector<float> &v2,
           DisType type) {
  unsigned int size = v1.size();
  __m128 sumVec = _mm_setzero_ps();
  float sum;

  switch (type) {
  case EucDis:
    for (unsigned i = 0; i < size; i += 4) {
      __m128 v1Vec = _mm_loadu_ps(&v1[i]);
      __m128 v2Vec = _mm_loadu_ps(&v2[i]);
      __m128 diff = _mm_sub_ps(v1Vec, v2Vec);
      __m128 squared = _mm_mul_ps(diff, diff);
      sumVec = _mm_add_ps(sumVec, squared);
    }
    sumVec = _mm_hadd_ps(sumVec, sumVec);
    sumVec = _mm_hadd_ps(sumVec, sumVec);

    _mm_store_ss(&sum, sumVec);
    sum = sqrt(sum);
    break;
  default:
    break;
  }
  return sum;
}

void load_centroids(vector<vector<vector<float>>> &centroid) {
  for (int i = 0; i < centroid.size(); i++) {
    string filename = "../centers" + to_string(i) + ".csv";
    ifstream file(filename);
    if (!file.is_open()) {
      cout << "fail to open" << filename << endl;
      return;
    }
    string line;
    while (getline(file, line)) {
      vector<float> row;
      stringstream ss(line);
      string cell;
      while (getline(ss, cell, ',')) {
        row.push_back(std::stof(cell));
      }
      centroid[i].push_back(row);
    }
    file.close();
  }
}
void load_clusters(vector<vector<int>> &labels) {
  for (int i = 0; i < 4; i++) {
    string filename = "../labels" + to_string(i) + ".csv";
    ifstream file(filename);
    if (!file.is_open()) {
      cout << "file to open" << filename << endl;
      return;
    }
    string line;
    while (getline(file, line)) {
      int val = stof(line);
      labels[i].emplace_back(val);
    }
    file.close();
  }
}
int find_closed(vector<float> v, vector<vector<float>> centroids) {
  float min = MAXFLOAT;
  int id = -1;
  for (int i = 0; i < centroids.size(); i++) {
    float dis = dist(v, centroids[i], EucDis);
    if (dis < min) {
      min = dis;
      id = i;
    }
  }
  return id;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  load_ivecs_data(path, ori_data, num, dim);
  load_centroids(centroids);
  load_clusters(clusters);

  vector<double> times;
  for (int i = 100; i <= 10000; i += 100) {
    double t1, t2;
    double timeuse;
    int problem_size = i;
    t1 = MPI_Wtime();
    vector<vector<int>> id(problem_size, vector<int>(sub_data_size, -1));
#pragma omp parallel for
    for (int i = 0; i < problem_size; i++) {
      vector<float> v0(ori_data[i].begin(), ori_data[i].begin() + 32);
      vector<float> v1(ori_data[i].begin() + 32, ori_data[i].begin() + 64);
      vector<float> v2(ori_data[i].begin() + 64, ori_data[i].begin() + 96);
      vector<float> v3(ori_data[i].begin() + 96, ori_data[i].end());
      id[i][0] = find_closed(v0, centroids[0]);
      id[i][1] = find_closed(v1, centroids[1]);
      id[i][2] = find_closed(v2, centroids[2]);
      id[i][3] = find_closed(v3, centroids[3]);
    }
    t2 = MPI_Wtime();
    timeuse = t2 - t1;
    printf("%f\n", timeuse);
    times.push_back(timeuse);
  }
  ofstream of("time.txt", ios::out);
  for (int i = 0; i < times.size(); i++) {
    of << times[i] << endl;
  }
  of.close();
  MPI_Finalize();
  return 0;
}