def compute_tp_gnn_loss(
    positive_scores,
    negative_scores,
    embeddings,
    positive_sources,
    positive_targets,
    negative_sources,
    negative_targets,
    source_batch_size,
    epoch,
    max_epochs,
    graph_scores,
    graph_sources,
    graph_targets,
    graph_net_ids,
    graph_weights,
    net_weights,
    node_features,
    node_areas,
):
    """Reproduce the maintained random-walk negative-sampling objective."""
    return -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Mechanism: deterministic area-quantile cut on the mean-center discriminant after area-weighted Lloyd refinement; Decision: the tier boundary is set by cumulative physical area at balance_target instead of nearest-center whole-group rounding; Prediction: this reorders only groups near the area median and should improve combined timing when the parent rounding left the final partition off the area target; Falsifier: matched-case combined score fails to improve while labels and timing-weighted cut remain unchanged."""
    node_count = embeddings.shape[0]
    centered_embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)
    dimension_variance = (centered_embeddings * centered_embeddings).mean(axis=0)
    initial_projection = (embeddings * dimension_variance).sum(axis=1)

    first_index = int(np.argmin(initial_projection))
    second_index = int(np.argmax(initial_projection))
    first_center = embeddings[first_index]
    second_center = embeddings[second_index]

    if first_index == second_index:
        first_center = embeddings[0]
        second_center = embeddings[node_count - 1]

    for iteration in range(max_iterations):
        first_distance = ((embeddings - first_center) * (embeddings - first_center)).sum(axis=1)
        second_distance = ((embeddings - second_center) * (embeddings - second_center)).sum(axis=1)
        high_side = (second_distance < first_distance).astype(np.int64)

        low_weights = node_areas * (1 - high_side)
        high_weights = node_areas * high_side
        low_total = low_weights.sum()
        high_total = high_weights.sum()

        low_mean = (embeddings * low_weights[:, None]).sum(axis=0) / (low_total + 1e-12)
        high_mean = (embeddings * high_weights[:, None]).sum(axis=0) / (high_total + 1e-12)

        first_center = np.where(low_total > 0, low_mean, first_center)
        second_center = np.where(high_total > 0, high_mean, second_center)

    discriminant = second_center - first_center
    final_projection = (embeddings * discriminant).sum(axis=1)

    sorted_order = np.argsort(final_projection)
    cumulative_area = np.cumsum(node_areas[sorted_order])
    target_area = node_areas.sum() * balance_target
    cut_position = int(np.argmin(np.abs(cumulative_area - target_area))) + 1
    lower_count = max(1, min(cut_position, node_count - 1))

    rank_order = np.argsort(np.argsort(final_projection))
    labels = (rank_order < lower_count).astype(np.int64)
    return labels
