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
    """Replace the parent's area-weighted reference k-means with a custom
    bounded two-means whose centers are unweighted embedding means and whose
    area balance is enforced by cumulative-area capacity rounding. The change
    keeps area only as a capacity constraint, reducing distortion of cluster
    geometry by very large groups while still targeting the same area split.
    Expected measurable effect: a different handoff partition with no larger
    closure movement and improved WNS/TNS on paired seeds. The hypothesis is
    invalidated if the partition is mostly unchanged, if closure moves most
    labels, or if timing fails to improve on the paired cases."""
    n_nodes = int(embeddings.shape[0])
    areas = node_areas
    total_area = float(node_areas.sum())
    target_area = balance_target * total_area

    mean_vector = embeddings.mean(axis=0)
    dist_mean = ((embeddings - mean_vector) ** 2).sum(axis=1)
    first_index = int(np.argmax(dist_mean))
    center_top = embeddings[first_index]

    dist_first = ((embeddings - center_top) ** 2).sum(axis=1)
    exclude_first = np.where(np.arange(n_nodes) == first_index, -1.0, dist_first)
    second_index = int(np.argmax(exclude_first))
    center_bottom = embeddings[second_index]

    dist_top = ((embeddings - center_top) ** 2).sum(axis=1)
    dist_bottom = ((embeddings - center_bottom) ** 2).sum(axis=1)
    scores = dist_bottom - dist_top
    order = np.argsort(-scores)
    area_sorted = areas[order]
    cum_area = np.cumsum(area_sorted)
    prefix_error = (cum_area - target_area) ** 2
    best_k = int(np.argmin(prefix_error)) + 1
    cut_k = min(max(best_k, 1), n_nodes - 1)
    rank = np.argsort(order)
    labels = np.where(rank < cut_k, 1, 0)

    for iteration in range(max_iterations):
        top_count = labels.sum()
        bottom_labels = 1 - labels
        bottom_count = bottom_labels.sum()

        center_top = (embeddings * labels[:, None]).sum(axis=0) / max(float(top_count), 1.0)
        center_bottom = (embeddings * bottom_labels[:, None]).sum(axis=0) / max(float(bottom_count), 1.0)

        dist_top = ((embeddings - center_top) ** 2).sum(axis=1)
        dist_bottom = ((embeddings - center_bottom) ** 2).sum(axis=1)
        scores = dist_bottom - dist_top
        order = np.argsort(-scores)
        area_sorted = areas[order]
        cum_area = np.cumsum(area_sorted)
        prefix_error = (cum_area - target_area) ** 2
        best_k = int(np.argmin(prefix_error)) + 1
        cut_k = min(max(best_k, 1), n_nodes - 1)
        rank = np.argsort(order)
        labels = np.where(rank < cut_k, 1, 0)

    return labels
