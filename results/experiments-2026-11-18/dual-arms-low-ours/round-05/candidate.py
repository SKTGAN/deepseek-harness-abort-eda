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
    """Diversify vs parent (decoder): replace the frozen reference two-way call
    with one custom bounded loop that alternates area-weighted center updates
    with a margin-sorted cumulative-area reassignment (balanced k-means seeded
    by a deterministic projection split). This is a different mechanism from
    the reference's single distance-preference ranking and from the earlier
    capacity-sweep attempt: it lets the discrete partition equilibrate across
    iterations while keeping the top-area share pinned near the balance target.

    Expected effect: a different discrete top/bottom assignment that better
    respects physical-area balance without abandoning embedding geometry.

    Invalidating evidence: partition_change_vs_initial stays at the initial
    assignment, or final timing is unchanged on identical assignments.
    """
    n = embeddings.shape[0]
    d = embeddings.shape[1]
    areas = node_areas * 1.0
    total_area = float(areas.sum())
    target_area = float(balance_target) * total_area

    axis_first = int(seed) % d
    axis_second = ((int(seed) // 7) + 1) % d
    proj = embeddings[:, axis_first] + 0.6180339887498949 * embeddings[:, axis_second]

    order_proj = np.argsort(proj)
    cum_proj = np.cumsum(areas[order_proj])
    split_proj = int((cum_proj < target_area).sum())
    split_proj = int(np.clip(split_proj, 1, n - 1))
    rank_proj = np.argsort(np.argsort(proj))
    labels = (rank_proj >= split_proj).astype(np.int64)

    for step_idx in range(max_iterations):
        mask_bottom = labels == 0
        mask_top = labels == 1
        w_bottom = float(areas[mask_bottom].sum())
        w_top = float(areas[mask_top].sum())
        c_bottom = (
            embeddings[mask_bottom] * areas[mask_bottom, None]
        ).sum(axis=0) / max(w_bottom, 1e-12)
        c_top = (
            embeddings[mask_top] * areas[mask_top, None]
        ).sum(axis=0) / max(w_top, 1e-12)
        dist_bottom = ((embeddings - c_bottom) ** 2).sum(axis=1)
        dist_top = ((embeddings - c_top) ** 2).sum(axis=1)
        margin = dist_bottom - dist_top

        order_margin = np.argsort(margin)
        cum_margin = np.cumsum(areas[order_margin])
        split_margin = int((cum_margin < target_area).sum())
        split_margin = int(np.clip(split_margin, 1, n - 1))
        rank_margin = np.argsort(np.argsort(margin))
        labels = (rank_margin >= split_margin).astype(np.int64)

    return labels
