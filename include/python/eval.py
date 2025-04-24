def compute_recall_and_precision(index, ground_truth, queries, k):
  """
  Computes the recall and precision for top-k nearest neighbors.

  Parameters:
    index: The index object with a method `query` to find nearest neighbors.
         `index.query(query, k)` should return the top-k nearest neighbors for a query.
    ground_truth: A dictionary where keys are query indices and values are sets of ground truth neighbors.
    queries: A list of query points.
    k: The number of nearest neighbors to consider.

  Returns:
    recall: The average recall value as a float.
    precision: The average precision value as a float.
  """
  total_recall = 0.0
  total_precision = 0.0
  num_queries = len(queries)

  for i, query in enumerate(queries):
    # Get the ground truth neighbors for the current query
    true_neighbors = ground_truth.get(i, set())

    # Query the index for top-k neighbors
    predicted_neighbors = set(index.query(query, k))

    # Compute the intersection of predicted and true neighbors
    correct_predictions = len(predicted_neighbors & true_neighbors)

    # Compute recall for this query
    if len(true_neighbors) > 0:
      total_recall += correct_predictions / len(true_neighbors)

    # Compute precision for this query
    if len(predicted_neighbors) > 0:
      total_precision += correct_predictions / len(predicted_neighbors)

  # Average recall and precision over all queries
  recall = total_recall / num_queries if num_queries > 0 else 0.0
  precision = total_precision / num_queries if num_queries > 0 else 0.0
  return recall, precision