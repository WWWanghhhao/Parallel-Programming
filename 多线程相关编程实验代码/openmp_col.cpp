#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
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
int size = 100;
int sub_data_size = 4;
char path[1024] = "../data/sift_query.fvecs";
vector<vector<float>> ori_data;
vector<vector<vector<float>>> centroids(sub_data_size);
// vector<vector<int>> clusters(sub_data_size);

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

int main() {

  load_ivecs_data(path, ori_data, num, dim);
  load_centroids(centroids);
  vector<double> times(100, 0);
  for (int k = 0; k < 10; k++) {
    for (int i = 100; i <= 10000; i += 100) {
      struct timeval t1, t2;
      double timeuse;
      int problem_size = i;
      gettimeofday(&t1, NULL);
      vector<vector<int>> id(problem_size, vector<int>(sub_data_size, -1));
      for (int j = 0; j < 4; j++) {
#pragma omp parallel for
        for (int k = 0; k < id.size(); k++) {
          auto v = vector<float>(ori_data[k].begin() + j * 32,
                                 ori_data[k].begin() + (j + 1) * 32);
          id[k][j] = find_closed(v, centroids[j]);
        }
      }
      gettimeofday(&t2, NULL);
      timeuse = (t2.tv_sec - t1.tv_sec) +
                (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
      printf("%.6f\n", timeuse);
      times[i / 100 - 1] += timeuse;
    }
  }
  ofstream ofs("time.txt", ios::out);
  for (int i = 0; i < times.size(); i++) {
    ofs << times[i] / double(10) << endl;
  }
  ofs.close();
  return 0;
}
// if (i == 10000) {
//   cout << "=============\n";
//   ofstream ofss("res-omp.txt", ios::out);
//   for (int i = 0; i < id.size(); i++) {
//     ofss << id[i][0] << " " << id[i][1] << " " << id[i][2] << " "
//          << id[i][3] << endl;
//   }
//   ofss.close();
// }