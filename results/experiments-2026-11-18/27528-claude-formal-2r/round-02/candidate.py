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
    """Replace the parent reference-helper call with one bounded custom two-center Lloyd loop.

    The change relative to the parent decoder is a seeded anchor plus farthest-anchor
    initialization followed by physical-area-weighted center updates and deterministic
    whole-group nearest-distance labels. Expected measurable effect: it can move groups
    across the partition boundary where the parent remained mostly unchanged, giving a
    chance for TNS improvement with WNS roughly neutral. The hypothesis is invalidated if
    changed-label pairs show no combined-score gain or if the largest regression repeats.
    """
    points = embeddings
    areas = node_areas
    num_nodes = points.shape[0]

    anchor_a = int(seed) % num_nodes
    center_a = points[anchor_a]
    seed_dist_a = np.sum((points - center_a) ** 2, axis=1)
    anchor_b = int(np.argmax(seed_dist_a))
    center_b = points[anchor_b]

    dist_a = np.sum((points - center_a) ** 2, axis=1)
    dist_b = np.sum((points - center_b) ** 2, axis=1)
    labels_out = np.where(dist_b < dist_a, 1, 0)

    for iteration in range(max_iterations):
        dist_a = np.sum((points - center_a) ** 2, axis=1)
        dist_b = np.sum((points - center_b) ** 2, axis=1)
        labels_out = np.where(dist_b < dist_a, 1, 0)

        count_a = int(np.sum(labels_out == 0))
        count_b = num_nodes - count_a

        if count_a == 0:
            nearest_a = int(np.argmin(dist_a))
            labels_out = np.where(np.arange(num_nodes) == nearest_a, 0, labels_out)
        if count_b == 0:
            nearest_b = int(np.argmin(dist_b))
            labels_out = np.where(np.arange(num_nodes) == nearest_b, 1, labels_out)

        mask_a = labels_out == 0
        mask_b = labels_out == 1
        weight_a = areas * mask_a
        weight_b = areas * mask_b
        den_a = np.sum(weight_a)
        den_b = np.sum(weight_b)

        if den_a > 0:
            center_a = np.sum(points * weight_a[:, None], axis=0) / den_a
        if den_b > 0:
            center_b = np.sum(points * weight_b[:, None], axis=0) / den_b

    return labels_out
