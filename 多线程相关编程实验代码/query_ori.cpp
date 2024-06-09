#include <algorithm>
#include <bits/types/struct_timeval.h>
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
vector<vector<vector<float>>> clusters_dis(sub_data_size);
vector<vector<int>> id(10000);

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
  string file_name = "new_res.txt";
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

void compute_id_dis(vector<vector<vector<float>>> &dis,
                    vector<vector<vector<float>>> centroid) {
  for (int i = 0; i < sub_data_size; i++) {
    dis[i].resize(256);
    for (int j = 0; j < 256; j++) {
      dis[i][j].resize(j + 1);
      for (int k = 0; k < j; k++) {
        float distance = dist(centroid[i][j], centroid[i][k], EucDis);
        dis[i][j][k] = distance;
      }
    }
  }
}
void compute_topK(vector<int> &query_id, vector<vector<int>> cluster,
                  vector<vector<vector<float>>> &dis, int k) {
  priority_queue<pair<float, int>> q;
  float distance = 0;
  // ofstream ofsss("dis.txt", ios::out);
  for (int i = 0; i < cluster.size(); i++) {
    if (q.size() < k) {
      distance = 0;
      for (int j = 0; j < query_id.size(); j++) {
        if (query_id[j] == cluster[i][j]) {
          continue;
        }
        int min_id = query_id[j] < cluster[i][j] ? query_id[j] : cluster[i][j];
        int max_id = query_id[j] > cluster[i][j] ? query_id[j] : cluster[i][j];
        distance += dis[j][max_id][min_id];
      }
      q.push(pair<float, int>(distance, i));
    } else {
      distance = 0;
      for (int j = 0; j < query_id.size(); j++) {
        if (query_id[j] == cluster[i][j]) {
          continue;
        }
        int min_id = query_id[j] < cluster[i][j] ? query_id[j] : cluster[i][j];
        int max_id = query_id[j] > cluster[i][j] ? query_id[j] : cluster[i][j];
        distance += dis[j][max_id][min_id];
      }
      q.push(pair<float, int>(distance, i));
      q.pop();
    }
  }
  //   return q;
}

const int NUM = 7;
pthread_t threads[NUM];
vector<priority_queue<pair<float, int>>> tmpRes(NUM);

void *process(void *arg) {
  int threadId = *((int *)arg);
  vector<int> query_id;
  for (int i = 1; i <= 4; i++) {
    query_id.emplace_back(*((int *)arg + i));
  }
  int start = threadId * (1000000 / NUM);
  int end = (threadId == NUM - 1) ? 1000000 : start + (1000000 / NUM);
  priority_queue<pair<float, int>> q;
  float distance = 0;
  for (int i = start; i < end; i++) {
    distance = 0;
    for (int j = 0; j < query_id.size(); j++) {
      if (query_id[j] == clusters[i][j])
        continue;
      int min_id = query_id[j] < clusters[i][j] ? query_id[j] : clusters[i][j];
      int max_id = query_id[j] > clusters[i][j] ? query_id[j] : clusters[i][j];
      distance += clusters_dis[j][max_id][min_id];
    }
    q.push(make_pair(distance, i));
    if (q.size() >= 100) {
      q.pop();
    }
  }
  tmpRes[threadId] = q;
  pthread_exit(NULL);
}

int main() {
  string p = "time" + to_string(NUM) + ".txt";
  load_query_ids(id);
  load_centroids(centroids);
  load_clusters(clusters);

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
  vector<double> time;
  vector<vector<int>> right_ans(10000, vector<int>(100));
  load_ivecs_data("../data/sift_groundtruth.ivecs", right_ans, num, dim);

  for (int j = 500; j <= 10000; j += 500) {
    struct timeval t1, t2;
    double timeuse = 0;
    gettimeofday(&t1, NULL);

    for (int i = 0; i < j; i++) {
      for (int k = 0; k < NUM; k++) {
        int *args = new int[5];
        args[0] = k;
        args[1] = id[i][0];
        args[2] = id[i][1];
        args[3] = id[i][2];
        args[4] = id[i][3];
        pthread_create(&threads[k], NULL, process, args);
      }
      for (int k = 0; k < NUM; k++) {
        pthread_join(threads[k], NULL);
      }
      priority_queue<pair<float, int>> pq;
      for (int k = 0; k < NUM; k++) {
        while (!tmpRes[k].empty()) {
          if (pq.size() < 100) {
            pq.push(tmpRes[k].top());
            tmpRes[k].pop();
          } else {
            pq.push(tmpRes[k].top());
            tmpRes[k].pop();
            pq.pop();
          }
        }
      }
      tmpRes.clear();
    }

    gettimeofday(&t2, NULL);
    timeuse =
        (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("%.6f\n", timeuse);
    time.push_back(timeuse);
  }

  ofstream ofs(p, ios::out);
  for (int i = 0; i < time.size(); i++) {
    ofs << time[i] << endl;
  }
  ofs.close();
  return 0;
}
