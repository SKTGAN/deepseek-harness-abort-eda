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
    """Replace the frozen two-means decoder with an area-balanced principal-axis split.

    Relative to the parent, this keeps the same legal area-balanced top-die
    fill but ranks groups by a power-iteration projection of the centered
    embeddings instead of by distance to two area-weighted means.  The
    expected measurable effect is a different discrete partition boundary
    and therefore different final timing.  A result with unchanged labels on
    most cases or no timing movement would invalidate the hypothesis.
    """
    n_groups = len(node_areas)
    n_dims = embeddings.shape[1]
    total_area = np.sum(node_areas)
    weighted_mean = np.sum(embeddings * node_areas[:, None], axis=0) / max(total_area, 1e-12)
    centered = embeddings - weighted_mean[None, :]
    weighted_variances = np.sum(node_areas[:, None] * centered * centered, axis=0) / max(total_area, 1e-12)
    dim = int(np.argmax(weighted_variances))
    vec = np.where(np.arange(n_dims) == dim, 1.0, 0.0)
    for iteration in range(max_iterations):
        projection = np.sum(centered * vec[None, :], axis=1)
        next_vec = np.sum(centered * (node_areas * projection)[:, None], axis=0) / max(total_area, 1e-12)
        vec = next_vec / np.sqrt(np.sum(next_vec * next_vec) + 1e-12)
    scores = np.sum(centered * vec[None, :], axis=1)
    order = np.argsort(-scores)
    sorted_areas = node_areas[order]
    cumulative = np.cumsum(sorted_areas)
    target_area = total_area * balance_target
    k_index = int(np.argmin(np.abs(cumulative - target_area)))
    k = min(max(k_index + 1, 1), max(n_groups - 1, 1))
    ranks = np.argsort(np.argsort(-scores))
    labels = (ranks < k).astype(int)
    return labels
