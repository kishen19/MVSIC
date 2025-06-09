#pragma once

#include "csvfile.h"
#include "mvc/utils/point_cloud_set.h"
#include "parlay/primitives.h"
#include "search_params.h"

namespace mvivf {
struct Stats {
	double QPS_seq = 0.0;
	double QPS_par = 0.0;
	double avg_cmps = 0.0;
	double recall_1_k = 0.0;
	double recall_k_k = 0.0;

	Stats() {}

	Stats(double QPS_seq, double QPS_par, double avg_cmps, double recall_1_k,
				double recall_k_k)
			: QPS_seq(QPS_seq), QPS_par(QPS_par), avg_cmps(avg_cmps),
				recall_1_k(recall_1_k), recall_k_k(recall_k_k) {}
};
} // namespace mvivf

double compute_recall(
		const parlay::sequence<parlay::sequence<std::pair<size_t, float>>> &pred,
		const parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>> &gt,
		size_t k, size_t k_gt) {
	if (k_gt > gt[0].size()) {
		std::cerr << "Not enough gt values" << std::endl;
		exit(-1);
	} else if (k > pred[0].size()) {
		std::cout << "Not enough pred values: " << pred[0].size() << " out of " << k
							<< std::endl;
	}
	parlay::internal::timer t;
	auto ind_recall =
			parlay::sequence<double>::from_function(pred.size(), [&](size_t i) {
				std::unordered_set<size_t> out_set;
				for (size_t j = 0; j < pred[i].size(); j++) {
					auto [id, dist] = pred[i][j];
					out_set.insert(id);
				}
				size_t correct = 0;
				for (size_t j = 0; j < k_gt; j++) {
					auto [dist, id] = gt[i][j];
					if (out_set.find(id) != out_set.end()) {
						correct++;
					}
				}
				// Dealing with duplicates and near duplicates: fine to return
				// any of the (near) duplicates of the last point
				float last_dist = gt[i][k_gt - 1].first;
				for (size_t j = k_gt; j < gt[i].size(); j++) {
					auto [dist, id] = gt[i][j];
					if (std::abs(dist - last_dist) < 1e-6) {
						if (out_set.find(id) != out_set.end()) {
							correct++;
						}
					} else {
						break;
					}
				}
				return static_cast<double>(correct) / static_cast<double>(k_gt);
			});
	auto val = parlay::reduce(ind_recall);
	return parlay::reduce(ind_recall) / static_cast<double>(ind_recall.size());
}

template <typename Index, typename ChPoint, typename GT>
mvivf::Stats compute_stats(Index &index, const PointCloudSet<ChPoint> &points,
													 const PointCloudSet<ChPoint> &query_points,
													 const GT &gt, const mvivf::SearchParams &params) {
	parlay::internal::timer t;
	size_t k = params.k;
	double query_time_seq = 0.0;
	auto pred = parlay::sequence<parlay::sequence<std::pair<size_t, float>>>(
			query_points.size());
	auto cmps = parlay::sequence<size_t>::uninitialized(query_points.size());
	for (size_t i = 0; i < query_points.size(); i++) {
		t.start();
		auto [results, dist_cmps_i] = index.search(query_points[i], points, params);
		t.stop();
		query_time_seq += t.total_time();
		t.reset();
		pred[i] = results;
		cmps[i] = dist_cmps_i;
	}
	t.start();
	parlay::parallel_for(0, query_points.size(), [&](size_t i) {
		auto [results, dist_cmps_i] = index.search(query_points[i], points, params);
		pred[i] = results;
		assert(cmps[i] == dist_cmps_i);
		cmps[i] = dist_cmps_i;
	});
	t.stop();
	double query_time_par = t.total_time();
	t.reset();
	double QPS_seq = query_points.size() / query_time_seq;
	double QPS_par = query_points.size() / query_time_par;
	double avg_cmps = (double)parlay::reduce(cmps) / (double)cmps.size();
	double recall_1_k = compute_recall(pred, gt, k, 1);
	double recall_k_k = compute_recall(pred, gt, k, k);
	return mvivf::Stats(QPS_seq, QPS_par, avg_cmps, recall_1_k, recall_k_k);
}

template <typename Index, typename ChPoint, typename GT>
mvivf::Stats
compute_stats_gold(Index &index, const PointCloudSet<ChPoint> &points,
									 const PointCloudSet<ChPoint> &query_points, const GT &gt,
									 const mvivf::SearchParams &params) {
	parlay::internal::timer t;
	size_t k = params.k;
	// Identify indices with at least 1 gt value
	auto atleast_1 = parlay::delayed_seq<size_t>(
			gt.size(), [&](size_t i) { return gt[i].size() > 0; });
	auto ind_atleast_1 = parlay::pack_index(atleast_1);
	std::cout << "At least 1: " << ind_atleast_1.size() << std::endl;
	auto pred_1 = parlay::sequence<parlay::sequence<std::pair<size_t, float>>>(
			ind_atleast_1.size());
	auto gt_1 = parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>>::
			from_function(ind_atleast_1.size(), [&](size_t id) {
				size_t i = ind_atleast_1[id];
				return gt[i];
			});
	auto cmps_1 = parlay::sequence<size_t>::uninitialized(ind_atleast_1.size());
	double query_time_seq = 0.0;
	for (size_t id = 0; id < ind_atleast_1.size(); id++) {
		size_t i = ind_atleast_1[id];
		t.start();
		auto [results, dist_cmps_i] = index.search(query_points[i], points, params);
		t.stop();
		query_time_seq += t.total_time();
		t.reset();
		pred_1[id] = results;
		cmps_1[id] = dist_cmps_i;
	}
	t.start();
	parlay::parallel_for(0, ind_atleast_1.size(), [&](size_t id) {
		size_t i = ind_atleast_1[id];
		auto [results, dist_cmps_i] = index.search(query_points[i], points, params);
		pred_1[id] = results;
		assert(cmps_1[id] == dist_cmps_i);
		cmps_1[id] = dist_cmps_i;
	});
	t.stop();
	double query_time_par = t.total_time();
	t.reset();
	// for (size_t i=0; i<10; ++i){
	//   std::cout << "gt: " << gt_1[i][0].second << std::endl;
	//   std::cout << "distance: " <<
	//   query_points[ind_atleast_1[i]].distance(points[gt_1[i][0].second]) <<
	//   std::endl; std::cout << "pred: "; for (size_t j=0; j<10; j++){
	//     std::cout << "(" << pred_1[i][j].first << ", " << pred_1[i][j].second
	//     << ") ";
	//   }
	//   std::cout << std::endl;
	// }
	double QPS_seq = ind_atleast_1.size() / query_time_seq;
	double QPS_par = ind_atleast_1.size() / query_time_par;
	double avg_cmps = (double)parlay::reduce(cmps_1) / (double)cmps_1.size();
	double recall_1_k = compute_recall(pred_1, gt_1, k, 1);
	return mvivf::Stats(QPS_seq, QPS_par, avg_cmps, recall_1_k, 0.0);
}

inline void write_to_csv(const std::string csv_filename,
												 const parlay::sequence<mvivf::Stats> &results,
												 const parlay::sequence<mvivf::SearchParams> &params) {
	assert(results.size() == params.size());
	csvfile csv(csv_filename);
	csv << "k"
			<< "nprobes"
			<< "QPS_seq"
			<< "QPS_par"
			<< "Avg Cmps"
			<< "Recall 1@k"
			<< "Recall k@k" << endrow;
	for (size_t i = 0; i < results.size(); ++i) {
		csv << params[i].k << params[i].nprobes << results[i].QPS_seq
				<< results[i].QPS_par << results[i].avg_cmps << results[i].recall_1_k
				<< results[i].recall_k_k << endrow;
	}
	csv << endrow;
}

// TODO: batch queries

template <typename Index, typename ChPoint, typename GT>
void search_all(Index &index, const PointCloudSet<ChPoint> &base_points,
								const PointCloudSet<ChPoint> &query_points, GT &gt,
								const char *res_file,
								const parlay::sequence<mvivf::SearchParams> &params) {
	parlay::sequence<mvivf::Stats> results;
	for (size_t i = 0; i < params.size(); ++i) {
		auto result =
				compute_stats(index, base_points, query_points, gt, params[i]);
		results.push_back(result);
	}
	write_to_csv(std::string(res_file), results, params);
}

template <typename Index, typename ChPoint, typename GT>
void search_all(Index &index, const PointCloudSet<ChPoint> &base_points,
								const PointCloudSet<ChPoint> &query_points, GT &gt,
								const char *res_file, const mvivf::SearchParams &params) {
	search_all(index, base_points, query_points, gt, res_file, {params});
}

template <typename Index, typename ChPoint, typename GT>
void search_all_gold(Index &index, const PointCloudSet<ChPoint> &base_points,
										 const PointCloudSet<ChPoint> &query_points, GT &gt,
										 const char *res_file,
										 const parlay::sequence<mvivf::SearchParams> &params) {
	parlay::sequence<mvivf::Stats> results;
	for (size_t i = 0; i < params.size(); ++i) {
		auto result =
				compute_stats_gold(index, base_points, query_points, gt, params[i]);
		results.push_back(result);
	}
	write_to_csv(std::string(res_file), results, params);
}

template <typename Index, typename ChPoint, typename GT>
void search_all_gold(Index &index, const PointCloudSet<ChPoint> &base_points,
										 const PointCloudSet<ChPoint> &query_points, GT &gt,
										 const char *res_file, const mvivf::SearchParams &params) {
	search_all_gold(index, base_points, query_points, gt, res_file, {params});
}

parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>>
ReadGT(std::string &file_path, size_t num_points) {
	std::ifstream file(file_path, std::ios::binary | std::ios::in);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file for reading: " + file_path);
	}

	int num_neighbors = 0;
	file.read(reinterpret_cast<char *>(&num_neighbors), sizeof(num_neighbors));

	parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>> result(
			num_points);
	for (int i = 0; i < num_points; ++i) {
		parlay::sequence<std::pair<float, uint32_t>> neighbors(num_neighbors);
		file.read(reinterpret_cast<char *>(neighbors.data()),
							num_neighbors * sizeof(std::pair<float, uint32_t>));
		result[i] = std::move(neighbors);
	}
	file.close();
	return result;
}

parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>>
ReadGoldGT(std::string &file_path, size_t num_points) {
	std::ifstream file(file_path, std::ios::binary | std::ios::in);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file for reading: " + file_path);
	}
	size_t num_offsets;
	size_t num_gt_entries;
	// Read sizes
	file.read(reinterpret_cast<char *>(&num_offsets), sizeof(num_offsets));
	file.read(reinterpret_cast<char *>(&num_gt_entries), sizeof(num_gt_entries));
	std::vector<size_t> offsets(num_offsets);
	std::vector<uint32_t> ground_truth(num_gt_entries);
	// Read offset and ground truth data
	file.read(reinterpret_cast<char *>(offsets.data()),
						num_offsets * sizeof(size_t));
	file.read(reinterpret_cast<char *>(ground_truth.data()),
						num_gt_entries * sizeof(uint32_t));
	file.close();
	// for (size_t i=0; i<offsets.size(); i++){
	//   std::cout << offsets[i] << " ";
	// }
	// std::cout << std::endl;
	// for (size_t i=0; i<ground_truth.size(); i++){
	//   std::cout << ground_truth[i] << " ";
	// }
	// std::cout << std::endl;
	parlay::sequence<parlay::sequence<std::pair<float, uint32_t>>> result(
			num_points);
	parlay::parallel_for(0, num_points, [&](size_t i) {
		size_t start_index = offsets[i];
		size_t end_index = offsets[i + 1];
		auto neighbors = parlay::tabulate(end_index - start_index, [&](size_t j) {
			return std::make_pair(static_cast<float>(j),
														ground_truth[start_index + j]);
		});
		result[i] = std::move(neighbors);
	});
	return result;
}