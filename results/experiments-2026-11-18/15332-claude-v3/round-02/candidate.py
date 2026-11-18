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
    """Loss behavior is unchanged from the parent: maintained random-walk objective."""
    return -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Replace the parent reference-helper call with a custom area-targeted two-center loop.

    Change vs parent: deterministic farthest-point two-center initialization,
    bounded area-weighted center updates, then a final whole-supernode split by
    projection rank with cumulative physical area closest to ``balance_target``.
    Expected effect: move supernodes across the area-balanced boundary that the
    reference helper left unchanged, improving TNS more than WNS while protected
    closure repairs macro feasibility. Invalid if the returned partition is
    mostly unchanged from the parent or if timed WNS/TNS regress on valid pairs.
    """
    n_nodes = embeddings.shape[0]
    total_area = node_areas.sum() + 1e-12

    weighted_mean = (embeddings * node_areas[:, None]).sum(axis=0) / total_area
    delta_mean = embeddings - weighted_mean
    dist_mean = (delta_mean * delta_mean).sum(axis=1)
    first_index = int(np.argmax(dist_mean))
    center_a = embeddings[first_index]

    delta_first = embeddings - center_a
    dist_first = (delta_first * delta_first).sum(axis=1)
    second_index = int(np.argmax(dist_first))
    center_b = embeddings[second_index]

    for iteration_index in range(max_iterations):
        delta_a = embeddings - center_a
        delta_b = embeddings - center_b
        dist_a = (delta_a * delta_a).sum(axis=1)
        dist_b = (delta_b * delta_b).sum(axis=1)
        side_b = (dist_b < dist_a) * 1
        weight_a = node_areas * (1 - side_b)
        weight_b = node_areas * side_b
        sum_a = weight_a.sum()
        sum_b = weight_b.sum()
        center_a = (
            (embeddings * weight_a[:, None]).sum(axis=0) + center_a * 1e-12
        ) / (sum_a + 1e-12)
        center_b = (
            (embeddings * weight_b[:, None]).sum(axis=0) + center_b * 1e-12
        ) / (sum_b + 1e-12)

    direction = center_b - center_a
    midpoint = (center_a + center_b) * 0.5
    scores = ((embeddings - midpoint) * direction).sum(axis=1)

    order = np.argsort(scores)
    rank = np.argsort(order)
    area_sorted = node_areas[order]
    cumulative_area = np.cumsum(area_sorted)
    target_area = balance_target * total_area
    area_error = cumulative_area - target_area
    split_index = int(np.argmin(area_error * area_error))
    split_count = int(min(max(split_index + 1, 1), n_nodes - 1))

    labels = (rank >= n_nodes - split_count) * 1
    return labels
