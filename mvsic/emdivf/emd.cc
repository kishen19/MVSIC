#include "mvsic/emdivf/emd.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace emdivf {
namespace {

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct AssignmentWorkspace {
  Matrix products;
  std::vector<double> costs;
  std::vector<double> u, v, minv;
  std::vector<size_t> p, way;
  std::vector<uint8_t> used;

  void Resize(size_t n) {
    costs.resize(n * n);
    u.resize(n + 1);
    v.resize(n + 1);
    minv.resize(n + 1);
    p.resize(n + 1);
    way.resize(n + 1);
    used.resize(n + 1);
  }
};

double Assignment(const double* cost, size_t n, AssignmentWorkspace& workspace) {
  // Shortest augmenting path Hungarian algorithm, O(n^3).
  auto& u = workspace.u;
  auto& v = workspace.v;
  auto& minv = workspace.minv;
  auto& p = workspace.p;
  auto& way = workspace.way;
  auto& used = workspace.used;
  std::fill(u.begin(), u.end(), 0.0);
  std::fill(v.begin(), v.end(), 0.0);
  std::fill(p.begin(), p.end(), 0);
  for (size_t i = 1; i <= n; ++i) {
    p[0] = i;
    std::fill(minv.begin(), minv.end(), std::numeric_limits<double>::infinity());
    std::fill(used.begin(), used.end(), false);
    size_t j0 = 0;
    do {
      used[j0] = true;
      const size_t i0 = p[j0];
      double delta = std::numeric_limits<double>::infinity();
      size_t j1 = 0;
      for (size_t j = 1; j <= n; ++j) {
        if (used[j]) continue;
        const double cur = cost[(i0 - 1) * n + j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (size_t j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const size_t j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }
  double total = 0;
  for (size_t j = 1; j <= n; ++j) total += cost[(p[j] - 1) * n + j - 1];
  return total / static_cast<double>(n);
}

struct Edge {
  int to;
  int reverse;
  int capacity;
  double cost;
};

void AddEdge(std::vector<std::vector<Edge>>& graph, int from, int to, int capacity,
             double cost) {
  graph[from].push_back({to, static_cast<int>(graph[to].size()), capacity, cost});
  graph[to].push_back({from, static_cast<int>(graph[from].size()) - 1, 0, -cost});
}

double BalancedTransport(const std::vector<double>& costs, int na, int nb) {
  // Each A point supplies nb integer units and each B point consumes na.
  // The total transported mass is na*nb, equivalent to uniform probability
  // masses 1/na and 1/nb after dividing the objective by that total.
  const int source = 0, sink = na + nb + 1, nodes = sink + 1;
  const int total = na * nb;
  std::vector<std::vector<Edge>> graph(nodes);
  for (int i = 0; i < na; ++i) AddEdge(graph, source, 1 + i, nb, 0);
  for (int j = 0; j < nb; ++j) AddEdge(graph, 1 + na + j, sink, na, 0);
  for (int i = 0; i < na; ++i) {
    for (int j = 0; j < nb; ++j) {
      AddEdge(graph, 1 + i, 1 + na + j, total, costs[i * nb + j]);
    }
  }
  std::vector<double> potential(nodes, 0), distance(nodes);
  std::vector<int> previous_node(nodes), previous_edge(nodes);
  int sent = 0;
  double objective = 0;
  while (sent < total) {
    std::fill(distance.begin(), distance.end(), std::numeric_limits<double>::infinity());
    distance[source] = 0;
    using Node = std::pair<double, int>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;
    heap.push({0, source});
    while (!heap.empty()) {
      auto [current, node] = heap.top();
      heap.pop();
      if (current > distance[node] + 1e-12) continue;
      for (size_t e = 0; e < graph[node].size(); ++e) {
        const Edge& edge = graph[node][e];
        if (edge.capacity == 0) continue;
        const double next = current + edge.cost + potential[node] - potential[edge.to];
        if (next + 1e-12 < distance[edge.to]) {
          distance[edge.to] = next;
          previous_node[edge.to] = node;
          previous_edge[edge.to] = static_cast<int>(e);
          heap.push({next, edge.to});
        }
      }
    }
    if (!std::isfinite(distance[sink])) throw std::runtime_error("EMD transport is infeasible");
    for (int node = 0; node < nodes; ++node) {
      if (std::isfinite(distance[node])) potential[node] += distance[node];
    }
    int flow = total - sent;
    for (int node = sink; node != source; node = previous_node[node]) {
      flow = std::min(flow, graph[previous_node[node]][previous_edge[node]].capacity);
    }
    for (int node = sink; node != source; node = previous_node[node]) {
      Edge& forward = graph[previous_node[node]][previous_edge[node]];
      Edge& reverse = graph[node][forward.reverse];
      forward.capacity -= flow;
      reverse.capacity += flow;
      objective += static_cast<double>(flow) * forward.cost;
    }
    sent += flow;
  }
  return objective / static_cast<double>(total);
}

}  // namespace

double EmdCapped(const float* a, uint32_t na, const float* b, uint32_t nb,
                 uint32_t dim, double cutoff) {
  if (na == 0 || nb == 0 || dim == 0) throw std::invalid_argument("EMD requires nonempty clouds");
  if (std::isnan(cutoff) || cutoff < 0) throw std::invalid_argument("invalid EMD cutoff");
  if (cutoff == 0) return 0;
  if (static_cast<uint64_t>(na) * nb > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("EMD transport problem is too large");
  }
  Eigen::Map<const Matrix> left(a, na, dim), right(b, nb, dim);
  thread_local AssignmentWorkspace workspace;
  workspace.products.resize(na, nb);
  workspace.products.noalias() = left * right.transpose();
  const auto left_norms = left.rowwise().squaredNorm().eval();
  const auto right_norms = right.rowwise().squaredNorm().eval();
  workspace.Resize(std::max(na, nb));
  auto& costs = workspace.costs;
  costs.resize(static_cast<size_t>(na) * nb);
  const bool bounded = std::isfinite(cutoff);
  std::vector<double> column_min(bounded ? nb : 0, std::numeric_limits<double>::infinity());
  double row_sum = 0;
  for (uint32_t i = 0; i < na; ++i) {
    double row_min = std::numeric_limits<double>::infinity();
    for (uint32_t j = 0; j < nb; ++j) {
      const double squared = static_cast<double>(left_norms[i]) + right_norms[j] -
                             2.0 * workspace.products(i, j);
      const double cost = std::sqrt(std::max(0.0, squared));
      costs[static_cast<size_t>(i) * nb + j] = cost;
      if (bounded) {
        row_min = std::min(row_min, cost);
        column_min[j] = std::min(column_min[j], cost);
      }
    }
    row_sum += row_min;
  }
  if (bounded) {
    const double columns = std::accumulate(column_min.begin(), column_min.end(), 0.0);
    // Margin protects a cutoff lying at floating-point summation roundoff.
    if (std::max(row_sum / na, columns / nb) > cutoff + 1e-12) return cutoff;
  }
  const double exact = na == nb ? Assignment(costs.data(), na, workspace)
                                : BalancedTransport(costs, na, nb);
  return std::min(exact, cutoff);
}

double Emd(const float* a, uint32_t na, const float* b, uint32_t nb, uint32_t dim) {
  return EmdCapped(a, na, b, nb, dim, std::numeric_limits<double>::infinity());
}

TopKResult EmdTopKEqual(const float* query, uint32_t n,
                        const std::vector<const float*>& clouds, uint32_t dim, uint32_t k) {
  if (n == 0 || dim == 0 || clouds.empty() || k == 0 || k > clouds.size())
    throw std::invalid_argument("invalid exact EMD top-k arguments");
  thread_local AssignmentWorkspace workspace;
  Eigen::Map<const Matrix> left(query, n, dim);
  const auto left_norms = left.rowwise().squaredNorm().eval();
  const size_t matrix_size = static_cast<size_t>(n) * n;
  std::vector<double> all_costs(clouds.size() * matrix_size);
  std::vector<std::pair<double, uint32_t>> bounds(clouds.size());
  for (uint32_t cloud = 0; cloud < clouds.size(); ++cloud) {
    Eigen::Map<const Matrix> right(clouds[cloud], n, dim);
    workspace.products.resize(n, n);
    workspace.products.noalias() = left * right.transpose();
    const auto right_norms = right.rowwise().squaredNorm().eval();
    double row_sum = 0;
    std::vector<double> column_min(n, std::numeric_limits<double>::infinity());
    double* costs = all_costs.data() + static_cast<size_t>(cloud) * matrix_size;
    for (uint32_t i = 0; i < n; ++i) {
      double row_min = std::numeric_limits<double>::infinity();
      for (uint32_t j = 0; j < n; ++j) {
        const double squared = static_cast<double>(left_norms[i]) + right_norms[j] -
                               2.0 * workspace.products(i, j);
        const double distance = std::sqrt(std::max(0.0, squared));
        costs[static_cast<size_t>(i) * n + j] = distance;
        row_min = std::min(row_min, distance);
        column_min[j] = std::min(column_min[j], distance);
      }
      row_sum += row_min;
    }
    const double column_sum = std::accumulate(column_min.begin(), column_min.end(), 0.0);
    bounds[cloud] = {std::max(row_sum, column_sum) / n, cloud};
  }
  std::sort(bounds.begin(), bounds.end());
  workspace.Resize(n);
  using Candidate = std::pair<double, uint32_t>;
  std::priority_queue<Candidate> best;
  uint32_t assignments = 0;
  for (const auto& [bound, cloud] : bounds) {
    if (best.size() == k && bound > best.top().first) break;
    const double distance = Assignment(
        all_costs.data() + static_cast<size_t>(cloud) * matrix_size, n, workspace);
    ++assignments;
    const Candidate candidate{distance, cloud};
    if (best.size() < k) best.push(candidate);
    else if (candidate < best.top()) { best.pop(); best.push(candidate); }
  }
  std::vector<Candidate> sorted(best.size());
  for (size_t i = sorted.size(); i > 0; --i) { sorted[i - 1] = best.top(); best.pop(); }
  TopKResult result;
  result.assignments = assignments;
  result.indices.reserve(sorted.size());
  result.distances.reserve(sorted.size());
  for (const auto& [distance, index] : sorted) {
    result.indices.push_back(index);
    result.distances.push_back(distance);
  }
  return result;
}

}  // namespace emdivf
