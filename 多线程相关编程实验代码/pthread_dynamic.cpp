#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <pthread.h>
#include <random>
#include <semaphore.h>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>
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

void *processRange(void *arg) {
  int start = *((int *)arg);
  int end = *((int *)arg + 1);

  for (int i = start; i < end; i++) {
    vector<float> v0(ori_data[i].begin(), ori_data[i].begin() + 32);
    vector<float> v1(ori_data[i].begin() + 32, ori_data[i].begin() + 64);
    vector<float> v2(ori_data[i].begin() + 64, ori_data[i].begin() + 96);
    vector<float> v3(ori_data[i].begin() + 96, ori_data[i].end());

    id[i][0] = find_closed(v0, centroids[0]);
    id[i][1] = find_closed(v1, centroids[1]);
    id[i][2] = find_closed(v2, centroids[2]);
    id[i][3] = find_closed(v3, centroids[3]);
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
    // vector<pthread_t> threads(num_threads);
    pthread_t *threads = new pthread_t[num_threads];
    vector<int> thread_args(2 * num_threads);

    for (int i = 0; i < num_threads; i++) {
      int start = i * (problem_size / num_threads);
      int end = (i == num_threads - 1) ? problem_size
                                       : start + (problem_size / num_threads);
      thread_args[2 * i] = start;
      thread_args[2 * i + 1] = end;
      pthread_create(&threads[i], NULL, processRange, &thread_args[2 * i]);
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
  ofstream ofs("time.txt", ios::out);
  for (int i = 0; i < times.size(); i++) {
    ofs << times[i] << endl;
  }
  ofs.close();

  return 0;
}
