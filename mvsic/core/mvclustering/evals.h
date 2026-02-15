#pragma once

#include <vector>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

namespace mvsic {

template<typename Seq>
double computeARI(const Seq& labels_true, const Seq& labels_pred) {
  if (labels_true.size() != labels_pred.size()) {
    std::cerr << "Size of true labels and predicted labels must be the same." << std::endl;
    abort();
  }

  size_t n = labels_true.size();
  std::unordered_map<size_t, size_t> true_cluster_sizes;
  std::unordered_map<size_t, size_t> pred_cluster_sizes;
  std::unordered_map<std::pair<size_t, size_t>, size_t, boost::hash<std::pair<size_t, size_t>>>
      contingency_table;

  for (size_t i = 0; i < n; ++i) {
    true_cluster_sizes[labels_true[i]]++;
    pred_cluster_sizes[labels_pred[i]]++;
    contingency_table[{labels_true[i], labels_pred[i]}]++;
  }

  double sum_comb_c = 0.0;
  for (const auto& entry : contingency_table) {
    if (entry.second > 1) {
      sum_comb_c += entry.second * (entry.second - 1) / 2.0;
    }
  }

  double sum_comb_true = 0.0;
  for (const auto& entry : true_cluster_sizes) {
    if (entry.second > 1) {
      sum_comb_true += entry.second * (entry.second - 1) / 2.0;
    }
  }

  double sum_comb_pred = 0.0;
  for (const auto& entry : pred_cluster_sizes) {
    if (entry.second > 1) {
      sum_comb_pred += entry.second * (entry.second - 1) / 2.0;
    }
  }

  double expected_index = (sum_comb_true * sum_comb_pred) / (n * (n - 1) / 2.0);
  double max_index = (sum_comb_true + sum_comb_pred) / 2.0;
  double adjusted_index = (sum_comb_c - expected_index) / (max_index - expected_index);

  return adjusted_index;
}

template<typename Seq>
double computeNMI(const Seq& labels_true, const Seq& labels_pred) {
  if (labels_true.size() != labels_pred.size()) {
    std::cerr << "Size of true labels and predicted labels must be the same." << std::endl;
    abort();
  }

  size_t n = labels_true.size();
  std::unordered_map<size_t, size_t> true_cluster_sizes;
  std::unordered_map<size_t, size_t> pred_cluster_sizes;
  std::unordered_map<std::pair<size_t, size_t>, size_t, boost::hash<std::pair<size_t, size_t>>>
      contingency_table;

  for (size_t i = 0; i < n; ++i) {
    true_cluster_sizes[labels_true[i]]++;
    pred_cluster_sizes[labels_pred[i]]++;
    contingency_table[{labels_true[i], labels_pred[i]}]++;
  }

  double mutual_info = 0.0;
  for (const auto& entry : contingency_table) {
    size_t true_cluster_size = true_cluster_sizes[entry.first.first];
    size_t pred_cluster_size = pred_cluster_sizes[entry.first.second];
    mutual_info += entry.second *
                   std::log((entry.second * n) / (double)(true_cluster_size * pred_cluster_size));
  }
  mutual_info /= n;

  double entropy_true = 0.0;
  for (const auto& entry : true_cluster_sizes) {
    entropy_true -= entry.second * std::log(entry.second / (double)n);
  }
  entropy_true /= n;

  double entropy_pred = 0.0;
  for (const auto& entry : pred_cluster_sizes) {
    entropy_pred -= entry.second * std::log(entry.second / (double)n);
  }
  entropy_pred /= n;

  double normalized_mutual_info = 2.0 * mutual_info / (entropy_true + entropy_pred);

  return normalized_mutual_info;
}

template<typename Seq>
std::tuple<double, double> computeMetrics(const Seq& labels_true, const Seq& labels_pred) {
  double ari = computeARI(labels_true, labels_pred);
  double nmi = computeNMI(labels_true, labels_pred);

  std::cout << "Adjusted Rand Index (ARI): " << ari << std::endl;
  std::cout << "Normalized Mutual Information (NMI): " << nmi << std::endl;

  return std::make_tuple(ari, nmi);
}

}  // namespace mvsic