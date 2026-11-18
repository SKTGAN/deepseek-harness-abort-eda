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
    """Unchanged from parent: reproduce the maintained random-walk negative-sampling objective."""
    return -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Change vs parent: replace the single frozen helper call with a bounded custom
    area-weighted two-center k-means. Initialization is deterministic from the weighted
    embedding mean and farthest embedding; assignment is a cumulative-area balanced
    nonempty rounding of the distance-difference order. Expected measurable effect:
    move groups across the area-balanced boundary on cells that previously kept the
    start assignment, changing the final partition and potentially TNS. The hypothesis
    is invalidated if timing and partition remain identical to the parent, or if the
    new partition fails to improve TNS on the regressed ariane133 seed.
    """
    n = int(embeddings.shape[0])
    safe_areas = np.abs(node_areas) + 1e-12
    total_area = np.sum(safe_areas)
    target_area = balance_target * total_area

    weighted_sum = np.sum(embeddings * safe_areas[:, None], axis=0)
    mean_center = weighted_sum / total_area
    mean_dist = np.sum((embeddings - mean_center) ** 2, axis=1)
    first_index = int(np.argmax(mean_dist))
    center_one = embeddings[first_index]

    first_dist = np.sum((embeddings - center_one) ** 2, axis=1)
    second_index = int(np.argmax(first_dist))
    center_two = embeddings[second_index]

    for step in range(max_iterations):
        dist_one = np.sum((embeddings - center_one) ** 2, axis=1)
        dist_two = np.sum((embeddings - center_two) ** 2, axis=1)
        diff = dist_one - dist_two

        order = np.argsort(diff)
        cumulative = np.cumsum(safe_areas[order])
        cut = int(np.sum(cumulative < target_area))
        cut = max(1, min(n - 1, cut))
        rank = np.argsort(np.argsort(diff))
        labels = np.where(rank < cut, 1, 0)

        area_one = np.sum(safe_areas * labels)
        area_two = total_area - area_one
        center_one = np.sum(
            embeddings * (safe_areas * labels)[:, None], axis=0
        ) / (area_one + 1e-12)
        center_two = np.sum(
            embeddings * (safe_areas * (1 - labels))[:, None], axis=0
        ) / (area_two + 1e-12)

    dist_one = np.sum((embeddings - center_one) ** 2, axis=1)
    dist_two = np.sum((embeddings - center_two) ** 2, axis=1)
    diff = dist_one - dist_two
    order = np.argsort(diff)
    cumulative = np.cumsum(safe_areas[order])
    cut = int(np.sum(cumulative < target_area))
    cut = max(1, min(n - 1, cut))
    rank = np.argsort(np.argsort(diff))
    labels = np.where(rank < cut, 1, 0)
    return labels
