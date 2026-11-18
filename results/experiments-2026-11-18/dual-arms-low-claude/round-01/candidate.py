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
    """Change from the parent: replace the seeded helper call with a deterministic
    maximum-variance-axis initialization and area-balanced iterations that rank
    groups by relative squared-distance margin rather than absolute margin. The
    cumulative whole-group prefix stays inside the 0.47-0.53 top-area window when
    a feasible prefix exists. This directly changes which groups cross the
    area-balanced boundary; expect a different final label assignment and timing.
    If the assignment and timing match the parent on most cells, the hypothesis
    that relative-margin ordering reaches the discrete partition is invalid.
    """
    total_area = float(np.sum(node_areas))
    total_area_safe = max(total_area, 1e-12)
    target_share = max(0.47, min(0.53, float(balance_target)))
    target_area = target_share * total_area
    lower_area = 0.47 * total_area
    upper_area = 0.53 * total_area

    center = np.sum(embeddings * node_areas[:, None], axis=0) / total_area_safe
    centered = embeddings - center
    variance = np.sum(node_areas[:, None] * centered * centered, axis=0) / total_area_safe
    axis = int(np.argmax(variance))
    scores = centered[:, axis]

    order = np.argsort(-scores)
    sorted_areas = node_areas[order]
    cumulative = np.cumsum(sorted_areas)
    inside = (cumulative >= lower_area) & (cumulative <= upper_area)
    inside_float = inside * 1.0
    cost = np.abs(cumulative - target_area) + (1.0 - inside_float) * (2.0 * total_area_safe)
    count = int(np.argmin(cost)) + 1
    count = max(1, min(embeddings.shape[0] - 1, count))
    rank = np.argsort(np.argsort(-scores))
    labels = (rank < count) * 1

    top_center = np.sum(
        embeddings * (labels == 1)[:, None] * node_areas[:, None], axis=0
    ) / max(float(np.sum(node_areas * (labels == 1))), 1e-12)
    bottom_center = np.sum(
        embeddings * (labels == 0)[:, None] * node_areas[:, None], axis=0
    ) / max(float(np.sum(node_areas * (labels == 0))), 1e-12)

    for step in range(max_iterations):
        top_dist = np.sum((embeddings - top_center) ** 2, axis=1)
        bottom_dist = np.sum((embeddings - bottom_center) ** 2, axis=1)
        scores = (bottom_dist - top_dist) / (bottom_dist + top_dist + 1e-12)

        order = np.argsort(-scores)
        sorted_areas = node_areas[order]
        cumulative = np.cumsum(sorted_areas)
        inside = (cumulative >= lower_area) & (cumulative <= upper_area)
        inside_float = inside * 1.0
        cost = np.abs(cumulative - target_area) + (1.0 - inside_float) * (2.0 * total_area_safe)
        count = int(np.argmin(cost)) + 1
        count = max(1, min(embeddings.shape[0] - 1, count))
        rank = np.argsort(np.argsort(-scores))
        labels = (rank < count) * 1

        top_center = np.sum(
            embeddings * (labels == 1)[:, None] * node_areas[:, None], axis=0
        ) / max(float(np.sum(node_areas * (labels == 1))), 1e-12)
        bottom_center = np.sum(
            embeddings * (labels == 0)[:, None] * node_areas[:, None], axis=0
        ) / max(float(np.sum(node_areas * (labels == 0))), 1e-12)

    return labels
