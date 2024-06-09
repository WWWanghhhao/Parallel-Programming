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

vector<int> kmeans(const vector<vector<float>> &data, int k,
                   vector<vector<float>> &centroid) {
  auto centroids = initCentroids(data, k, 42);
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
// void load_clusters(vector<vector<int>> &clusters) {
//   vector<vector<int>> tmp(clusters);
//   for (int i = 0; i < clusters.size(); i++) {
//     string cluster_file = to_string(i) + "_cluster.txt";
//     cluster_file = "../" + cluster_file;
//     ifstream ifs(cluster_file, ios::in);
//     if (!ifs.is_open()) {
//       cerr << "Failed to open file: " << cluster_file << endl;
//       return;
//     }
//     int cluster_id;
//     while (ifs >> cluster_id) {
//       tmp[i].push_back(cluster_id);
//     }
//     ifs.close();
//   }
//   clusters = vector<vector<int>>(1000000, vector<int>(4));
//   auto row0 = tmp[0];
//   auto row1 = tmp[1];
//   auto row2 = tmp[2];
//   auto row3 = tmp[3];
//   for (int i = 0; i < 1000000; i++) {
//     clusters[i][0] = row0[i];
//     clusters[i][1] = row1[i];
//     clusters[i][2] = row2[i];
//     clusters[i][3] = row3[i];
//   }
// }

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

void encoding(int problem_size, vector<double> &times) {
  struct timeval t1, t2;
  double timeuse;
  gettimeofday(&t1, NULL);
  vector<vector<int>> id(problem_size, vector<int>(sub_data_size, -1));
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
  gettimeofday(&t2, NULL);
  timeuse =
      (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
  printf("%.6f seconds \n", timeuse);
  times.push_back(timeuse);
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
vector<int> compute_topK(vector<int> &query_id, vector<vector<int>> cluster,
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
  vector<int> ans;
  while (!q.empty()) {
    ans.emplace_back(q.top().second);
    q.pop();
  }
  reverse(ans.begin(), ans.end());
  return ans;
}

int main() {

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

  vector<vector<int>> right_ans(10000, vector<int>(100));
  load_ivecs_data("../data/sift_groundtruth.ivecs", right_ans, num, dim);

  // #pragma omp parallel for
  struct timeval t1, t2;
  vector<double> time;
  // vector<priority_queue<pair<float, int>>> tmpRes(12);

  for (int r = 500; r <= 10000; r += 500) {
    // double timeuse;
    gettimeofday(&t1, NULL);

    for (int i = 0; i < r; i++) {
      priority_queue<pair<float, int>> pq;
      for (int j = 0; j < clusters.size(); j++) {

        float distance = 0;
        for (int k = 0; k < 4; k++) {
          if (id[i][k] == clusters[j][k]) {
            continue;
          }
          int min_id = id[i][k] < clusters[j][k] ? id[i][k] : clusters[j][k];
          int max_id = id[i][k] > clusters[j][k] ? id[i][k] : clusters[j][k];
          distance += clusters_dis[k][max_id][min_id];
        }
        // int threadId = omp_get_thread_num();
        pq.push(pair<float, int>(distance, j));
        if (pq.size() >= 100) {
          pq.pop();
        }
      }

      // for (int k = 0; k < tmpRes.size(); k++) {
      //   while (!tmpRes[k].empty()) {
      //     pq.push(tmpRes[k].top());
      //     if (pq.size() >= 100)
      //       pq.pop();
      //     tmpRes[k].pop();
      //   }
      // }
    }

    gettimeofday(&t2, NULL);
    auto timeuse =
        (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
    cout << timeuse << endl;
    time.emplace_back(timeuse);
  }
  // vector<int> cnt(10000, 0);

  ofstream ofs("time.txt", ios::out);
  for (int i = 0; i < time.size(); i++) {
    ofs << time[i] << endl;
  }
  ofs.close();
  return 0;
}
