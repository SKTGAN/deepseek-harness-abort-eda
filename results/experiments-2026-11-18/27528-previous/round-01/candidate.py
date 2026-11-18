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
    """Parent delegates to reference_global_two_way. This version changes the
    decoder to a deterministic area-balanced embedding sweep refined by bounded
    capacity-constrained Lloyd updates: each step ranks nodes by distance
    preference and cuts at the cumulative physical-area target, rather than
    returning the frozen seeded two-means assignment directly. Expected
    measurable effect: a different legal discrete die boundary with a potentially
    lower timing-weighted cut. The hypothesis is invalidated if the final layer
    assignment and official timing are unchanged, or if closure repairs the
    sweep back to the parent partition without timing benefit.
    """
    local_embeddings = embeddings * 1.0
    local_areas = node_areas * 1.0
    local_n = int(local_embeddings.shape[0])
    total_area = float(np.sum(local_areas))
    target_area = float(balance_target) * total_area
    safe_total = max(total_area, 1e-12)

    area_mean = np.sum(local_embeddings * local_areas[:, None], axis=0) / safe_total
    deviation = np.sum((local_embeddings - area_mean) ** 2, axis=1)
    weighted_deviation = deviation * local_areas
    top_index = int(np.argmax(weighted_deviation))
    top_anchor = local_embeddings[top_index]

    distance_to_top = np.sum((local_embeddings - top_anchor) ** 2, axis=1)
    bottom_index = int(np.argmax(distance_to_top * local_areas))
    bottom_anchor = local_embeddings[bottom_index]

    dist_top_init = np.sum((local_embeddings - top_anchor) ** 2, axis=1)
    dist_bottom_init = np.sum((local_embeddings - bottom_anchor) ** 2, axis=1)
    score_init = dist_bottom_init - dist_top_init
    order_init = np.argsort(-score_init, kind="mergesort")
    rank_init = np.argsort(order_init, kind="mergesort")
    cum_init = np.cumsum(local_areas[order_init])
    cut_init = int(np.argmin(np.abs(cum_init - target_area))) + 1
    cut_init = min(max(cut_init, 1), local_n - 1)
    labels = np.where(rank_init < cut_init, 1, 0)

    for iteration in range(max_iterations):
        top_mask = labels > 0
        top_area = float(np.sum(local_areas * top_mask))
        bottom_area = safe_total - top_area
        top_center = np.sum(
            local_embeddings * (local_areas * top_mask)[:, None], axis=0
        ) / max(top_area, 1e-12)
        bottom_center = np.sum(
            local_embeddings * (local_areas * (1 - top_mask))[:, None], axis=0
        ) / max(bottom_area, 1e-12)

        dist_top = np.sum((local_embeddings - top_center) ** 2, axis=1)
        dist_bottom = np.sum((local_embeddings - bottom_center) ** 2, axis=1)
        score = dist_bottom - dist_top

        order = np.argsort(-score, kind="mergesort")
        rank = np.argsort(order, kind="mergesort")
        cum = np.cumsum(local_areas[order])
        cut = int(np.argmin(np.abs(cum - target_area))) + 1
        cut = min(max(cut, 1), local_n - 1)
        labels = np.where(rank < cut, 1, 0)

    return labels.astype(np.int64)
