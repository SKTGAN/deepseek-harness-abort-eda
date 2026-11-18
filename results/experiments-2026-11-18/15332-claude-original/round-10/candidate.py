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
    """Relative to the parent (random-walk contrastive loss only), add a
    timing-weighted star-edge attraction term
        sum_e w_e * softplus(-graph_scores_e) / sum_e w_e
    rescaled by (P+Q)/source_batch_size so its total gradient is comparable to
    the existing pair term and is strong enough to reorder groups at the
    area-balanced two-means boundary. Expected measurable effect: embeddings
    keep timing-heavy nets locally close, lowering the timing-weighted cut and
    shifting the official decoder's discrete top/bottom split, improving
    WNS/TNS. The hypothesis is invalidated if the final layer assignment is
    essentially unchanged or the timing-weighted cut does not fall.
    """
    pair_term = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    weight_total = graph_weights.sum()
    edge_mean = (graph_weights * F.softplus(-graph_scores)).sum() / (weight_total + 1e-6)
    pair_count = float(len(positive_scores)) + float(len(negative_scores))
    batch_scale = pair_count / max(1.0, float(source_batch_size))
    return pair_term + batch_scale * edge_mean


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Relative to the parent custom area-balanced two-means decoder, replace
    the distance-margin ranking by a seeded area-weighted variance-axis
    projection refined with an iterative Fisher between-group direction. Each
    bounded iteration ranks groups by their one-dimensional projection,
    chooses the top-die fill count by cumulative physical area near a clipped
    balance target, and updates the projection direction from the area-weighted
    top and bottom means. This changes the discrete boundary decision directly,
    because the projection order determines which groups cross the area-balanced
    cutoff. Expected measurable effect: a different legal split that improves
    the worst-case timing score, especially on bp-ng45, without sacrificing the
    strong ariane133 cells. The hypothesis is invalidated if the final layer
    assignment is identical to the parent decoder's or official timing does not
    improve.
    """
    n_nodes = int(embeddings.shape[0])
    n_dims = int(embeddings.shape[1])
    total_area = float(node_areas.sum())
    clipped_target = min(0.53, max(0.47, float(balance_target)))
    target_area = clipped_target * total_area
    lower_area = 0.47 * total_area
    upper_area = 0.53 * total_area

    mean_all = (node_areas[:, None] * embeddings).sum(axis=0) / max(total_area, 1e-12)
    centered = embeddings - mean_all
    weighted_sq = (node_areas[:, None] * centered * centered).sum(axis=0) / max(total_area, 1e-12)
    axis_index = int(np.argmax(weighted_sq))
    direction = np.where(np.arange(n_dims) == axis_index, 1.0, 0.0)
    labels = np.where(np.arange(n_nodes) < max(1, n_nodes // 2), 1, 0)

    for iteration in range(max_iterations):
        projection = (centered * direction[None, :]).sum(axis=1)
        score = projection - (np.arange(n_nodes) * 1e-12)
        order = np.argsort(-score)
        sorted_areas = node_areas[order]
        cumulative = np.cumsum(sorted_areas)
        below = np.where(cumulative < lower_area, lower_area - cumulative, 0.0)
        above = np.where(cumulative > upper_area, cumulative - upper_area, 0.0)
        selection_cost = np.abs(cumulative - target_area) + 1e6 * (below + above)
        count = int(np.argmin(selection_cost)) + 1
        count = max(1, min(n_nodes - 1, count))
        threshold = score[order[count - 1]]
        labels = np.where(score >= threshold, 1, 0)

        top_mask = labels == 1
        bottom_mask = labels == 0
        top_area = float(node_areas[top_mask].sum())
        bottom_area = float(node_areas[bottom_mask].sum())
        top_mean = (
            node_areas[top_mask, None] * centered[top_mask]
        ).sum(axis=0) / max(top_area, 1e-12)
        bottom_mean = (
            node_areas[bottom_mask, None] * centered[bottom_mask]
        ).sum(axis=0) / max(bottom_area, 1e-12)
        new_direction = top_mean - bottom_mean
        norm = float(np.sqrt((new_direction * new_direction).sum()))
        direction = np.where(norm > 1e-12, new_direction / max(norm, 1e-12), direction)

    return labels
