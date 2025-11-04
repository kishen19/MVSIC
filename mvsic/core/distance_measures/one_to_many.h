#pragma once

namespace mvsic {

std::vector<std::pair<uint32_t, float>> chamfer_ip_distance_one_to_many(
    const float* query_points, uint32_t n_query, const float* database_points, uint32_t n_database,
    uint32_t dim, uint32_t k);

}  // namespace mvsic