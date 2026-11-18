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
    """Area-balanced cosine reassignment of the frozen two-means labels.

    Parent returned the seeded area-balanced weighted two-means labels directly.
    This candidate keeps that single unchanged helper call for the two centers,
    then reassigns all groups by area-balanced ranking of the cosine-similarity
    difference between the two physical-area-weighted centers. Expected effect:
    assignments follow the L2-normalized dot-product geometry used by the loss,
    which can sharpen the learned communities and improve post-flow timing.
    Invalidation: if the decoded labels are rejected, final cut/timing does not
    improve versus the parent, or the paired physical score regresses.
    """
    labels = reference_global_two_way(
        embeddings, node_areas, seed, max_iterations, balance_target
    )

    n = embeddings.shape[0]
    if n < 2:
        return labels

    top_weights = labels * node_areas
    bottom_weights = (1 - labels) * node_areas
    top_den = np.sum(top_weights)
    bottom_den = np.sum(bottom_weights)
    if top_den <= 0.0 or bottom_den <= 0.0:
        return labels

    top_center = np.sum(embeddings * top_weights[:, None], axis=0) / top_den
    bottom_center = np.sum(embeddings * bottom_weights[:, None], axis=0) / bottom_den

    top_norm = np.sqrt(np.sum(top_center * top_center))
    bottom_norm = np.sqrt(np.sum(bottom_center * bottom_center))
    if top_norm <= 1e-12 or bottom_norm <= 1e-12:
        return labels

    top_dir = top_center / top_norm
    bottom_dir = bottom_center / bottom_norm
    score = (
        np.sum(embeddings * top_dir[None, :], axis=1)
        - np.sum(embeddings * bottom_dir[None, :], axis=1)
    )

    order = np.argsort(-score)
    ranks = np.argsort(order)
    sorted_areas = node_areas[order]
    cumulative_area = np.cumsum(sorted_areas)
    target_area = balance_target * np.sum(node_areas)
    k = int(np.argmin(np.abs(cumulative_area - target_area))) + 1
    if k < 1:
        k = 1
    if k > n - 1:
        k = n - 1

    return np.where(ranks < k, 1, 0)
