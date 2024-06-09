#include "/usr/local/mpich/include/mpi.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <pthread.h>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <utility>
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
vector<vector<int>> clusters(1000000, vector<int>(4));

void load_ivecs_data(const char *filename, vector<vector<int>> &results,
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
    int flag = 0;
    while (getline(file, line)) {
      int val = stof(line);
      labels[flag++][i] = val;
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

void load_query_ids(vector<vector<int>> &ids) {
  string file_name = "../多线程/new_res.txt";
  ifstream ifs(file_name);
  if (!ifs.is_open()) {
    cerr << "failed to open id file: " << file_name << endl;
    return;
  }
  for (int i = 0; i < ids.size(); i++) {
    string line;
    getline(ifs, line);
    istringstream iss(line);
    vector<int> id(4);
    for (int j = 0; j < 4; j++) {
      int val;
      if (!(iss >> val)) {
        cerr << "Error reading file: " << file_name << endl;
        return;
      }
      id[j] = val;
    }
    ids[i] = id;
  }
  ifs.close();
}

int main(int argc, char **argv) {

  vector<vector<int>> id(10000);
  load_query_ids(id);
  load_centroids(centroids);
  load_clusters(clusters);
  vector<vector<vector<float>>> clusters_dis(sub_data_size);
  for (int i = 0; i < sub_data_size; i++) {
    clusters_dis[i].resize(256);
    for (int j = 0; j < 256; j++) {
      clusters_dis[i][j].resize(j + 1);
      for (int k = 0; k < j; k++) {
        float distance = dist(centroids[i][j], centroids[i][k], EucDis);
        clusters_dis[i][j][k] = distance;
      }
    }
  }
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  vector<double> time;
  double t1, t2, timeuse;

  for (int r = 500; r <= 10000; r += 500) {
    // double timeuse;
    t1 = MPI_Wtime();
#pragma omp parallel for 
    for (int i = 0; i < r; i++) {
      vector<priority_queue<pair<float, int>>> pq(size);

      int problem_size = clusters.size();
      int start = rank * (problem_size / size);
      int end = (rank == size - 1) ? problem_size
                                   : (rank + 1) * (problem_size / size);

      for (int j = start; j < end; j++) {

        float distance = 0;
        for (int k = 0; k < 4; k++) {
          if (id[i][k] == clusters[j][k]) {
            continue;
          }
          int min_id = id[i][k] < clusters[j][k] ? id[i][k] : clusters[j][k];
          int max_id = id[i][k] > clusters[j][k] ? id[i][k] : clusters[j][k];
          distance += clusters_dis[k][max_id][min_id];
        }

        pq[rank].push(pair<float, int>(distance, j));
        if (pq[rank].size() >= 100) {
          pq[rank].pop();
        }
      }



    }

    t2 = MPI_Wtime();
    timeuse = t2 - t1;
    double global_time = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&timeuse, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      printf("%f\n", global_time);
      time.push_back(global_time);
    }
  }

  if (rank == 0) {
    ofstream ofs("time.txt", ios::out);
    for (int i = 0; i < time.size(); i++) {
      ofs << time[i] << endl;
    }
    ofs.close();
  }
  MPI_Finalize();
  return 0;
}
