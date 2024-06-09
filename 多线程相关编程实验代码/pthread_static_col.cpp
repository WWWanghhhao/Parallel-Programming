#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <pmmintrin.h>
#include <pthread.h>
#include <random>
#include <semaphore.h>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
#include <xmmintrin.h>
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
vector<vector<int>> id;
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
void load_centroids(vector<vector<vector<float>>> &centroids) {
  for (int i = 0; i < centroids.size(); i++) {
    string centroid_file = to_string(i) + "_centroid.txt";
    centroid_file = "../" + centroid_file;
    ifstream ifs(centroid_file, ios::in);
    if (!ifs.is_open()) {
      cerr << "Failed to open file: " << centroid_file << endl;
      return;
    }
    string line;
    while (getline(ifs, line)) {
      istringstream iss(line);
      float val;
      vector<float> centroid;
      while (iss >> val) {
        centroid.push_back(val);
      }
      centroids[i].push_back(centroid);
    }
    ifs.close();
  }
}
void load_clusters(vector<vector<int>> &clusters) {
  vector<vector<int>> tmp(clusters);
  for (int i = 0; i < clusters.size(); i++) {
    string cluster_file = to_string(i) + "_cluster.txt";
    cluster_file = "../" + cluster_file;
    ifstream ifs(cluster_file, ios::in);
    if (!ifs.is_open()) {
      cerr << "Failed to open file: " << cluster_file << endl;
      return;
    }
    int cluster_id;
    while (ifs >> cluster_id) {
      tmp[i].push_back(cluster_id);
    }
    ifs.close();
  }
  clusters = vector<vector<int>>(1000000, vector<int>(4));
  auto row0 = tmp[0];
  auto row1 = tmp[1];
  auto row2 = tmp[2];
  auto row3 = tmp[3];
  for (int i = 0; i < 1000000; i++) {
    clusters[i][0] = row0[i];
    clusters[i][1] = row1[i];
    clusters[i][2] = row2[i];
    clusters[i][3] = row3[i];
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

void *processColumn(void *arg) {
  int column_idx = *((int *)arg);

  for (int i = 0; i < id.size(); i++) {
    vector<float> v =
        vector<float>(ori_data[i].begin() + column_idx * 32,
                      ori_data[i].begin() + (column_idx + 1) * 32);

    id[i][column_idx] = find_closed(v, centroids[column_idx]);
  }

  pthread_exit(NULL);
}

int main() {
  load_ivecs_data(path, ori_data, num, dim);
  load_centroids(centroids);
  load_clusters(clusters);
  vector<double> times;

  for (int j = 100; j <= 10000; j += 100) {
    struct timeval t1, t2;
    double timeuse;
    int problem_size = j;
    gettimeofday(&t1, NULL);

    id = vector<vector<int>>(problem_size, vector<int>(sub_data_size, -1));

    int num_threads = 4;
    vector<pthread_t> threads(num_threads);
    vector<int> thread_args(num_threads);

    for (int i = 0; i < num_threads; i++) {
      thread_args[i] = i;
      pthread_create(&threads[i], NULL, processColumn, &thread_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
      pthread_join(threads[i], NULL);
    }

    gettimeofday(&t2, NULL);
    timeuse =
        (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("%.6f\n", timeuse);
    times.push_back(timeuse);
  }

  return 0;
}