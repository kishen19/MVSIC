// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
namespace graph_mining {

// Config type for GenerateFixedDimensionalEncoding function.
struct FixedDimensionalEncodingConfig {
  // Dimension of the input embeddings.
  int32_t dimension;

  // Number of independent repetitions for FDE generation.
  int32_t num_repetitions = 1;

  // Number of SimHash projections used to partition space in each repetition.
  // Is ignored if a partitioning method that is not "DEFAULT_SIMHASH" is
  // being used.
  int32_t num_simhash_projections;

  // Seed for the FDE generation process. Must be set to the same value for
  // query and document FDE generation to ensure coonsistency of the partitions
  // and projections used for queries and docuemnets.
  int32_t seed = 1;

  // How embeddings are added to the FDE. `DEFAULT_SUM` means that points are
  // summed into the respective locations in the FDE. `AVERAGE` means that an
  // average of all embeddings mapped to a partition in the FDE is taken.
  // Generally, Query FDE Encoodings will  use `DEFAULT_SUM`, whereas document
  // side FDE generation will use `AVERAGE`.
  enum EncodingType { DEFAULT_SUM, AVERAGE };

  EncodingType encoding_type = DEFAULT_SUM;

  // If a random projection `encoding_type` is being used, this is the
  // dimension to which points are reduced via random projections.
  int32_t projection_dimension;

  // The ProjectionType sets how the original embeddings are projected down to
  // `projection_dimension` dimensional space before being added to the FDE.
  // -- `DEFAULT_IDENTITY` means that no projection is set, and the original
  // embeddings are added to the FDE.
  // --  `AMS_SKETCH` a dense AMS sketch is used to project the data original
  // embeddings.
  enum ProjectionType { DEFAULT_IDENTITY, AMS_SKETCH };

  ProjectionType projection_type = DEFAULT_IDENTITY;

  // If true, the blocks of the output FDE corresponding to an empty partition
  // are filled with the coordinates of the point in the input point cloud
  // which is nearest to the partition. Here, the nearest point is defined as
  // the point which minimizes the number of disagreeing bits between its
  // SimHash sketch and the bits in the sketch corresponding to the partition.
  // This option throws an InvalidArgumentError if enabled when used with
  // query FDE generation.
  bool fill_empty_partitions = false;

  // If set, the final FDE is projected down to this dimension using a
  // random projection. The random projection is implemented using Count-Sketch
  // (i.e., sparse Johnson-Lindenstrauss projection) for efficiency purposes,
  // allowing the original FDE to be constructed with large dimension before
  // being projected down.
  int32_t final_projection_dimension = 0;
  bool has_final_projection_dimension() const { return final_projection_dimension > 0; }
};
}  // namespace graph_mining