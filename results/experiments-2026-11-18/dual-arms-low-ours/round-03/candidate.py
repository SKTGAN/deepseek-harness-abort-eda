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
    """Change vs parent (decoder): replace the frozen reference two-way k-means call with
    one custom bounded loop that assigns supernodes by a physical-area capacity sweep over
    the current two-center distance preference, then refreshes both centers as area-weighted
    means. Expected effect: the returned discrete die partition is a cumulative-area fill in
    preference order rather than the reference nearest-area rounding, so a different
    near-feasible cut enters the protected closure and the measured timing moves. Invalidating
    result: partition diagnostics stay at the initial assignment, or the joint timing score
    does not beat the reference on the matched cells."""
    n = int(embeddings.shape[0])
    total_area = float(node_areas.sum())
    half_area = 0.5 * (total_area + 1e-30)
    first = int(seed) % n
    second = (int(seed) * 7919 + n // 2) % n
    c_top = embeddings[first]
    c_bot = embeddings[second]

    d_top = ((embeddings - c_top) ** 2).sum(axis=1)
    d_bot = ((embeddings - c_bot) ** 2).sum(axis=1)
    pref = d_bot - d_top
    order = np.argsort(-pref)
    ranks = np.argsort(order)
    sorted_areas = node_areas[order]
    cum_before = np.cumsum(sorted_areas) - sorted_areas
    keep = (cum_before < half_area) & (ranks < n - 1)
    labels = np.where(keep[ranks], 1, 0)

    for step in range(max_iterations):
        d_top = ((embeddings - c_top) ** 2).sum(axis=1)
        d_bot = ((embeddings - c_bot) ** 2).sum(axis=1)
        pref = d_bot - d_top
        order = np.argsort(-pref)
        ranks = np.argsort(order)
        sorted_areas = node_areas[order]
        cum_before = np.cumsum(sorted_areas) - sorted_areas
        keep = (cum_before < half_area) & (ranks < n - 1)
        mask_top = keep[ranks]
        labels = np.where(mask_top, 1, 0)
        weights_top = node_areas * mask_top
        weights_bot = node_areas * (1.0 - mask_top)
        c_top = (embeddings * weights_top[:, None]).sum(axis=0) / (
            float(weights_top.sum()) + 1e-30
        )
        c_bot = (embeddings * weights_bot[:, None]).sum(axis=0) / (
            float(weights_bot.sum()) + 1e-30
        )
    return labels
