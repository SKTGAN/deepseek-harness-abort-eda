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
    """Mechanism: seeded area-weighted two-means with deterministic farthest anchors and a cumulative-area cut; Decision: each node's tier follows the nearer area-balanced center after ranking by squared-distance margin; Prediction: tighter embedding-coherent balanced groups reduce timing-weighted cut and improve combined timing; Falsifier: unchanged final weighted cut or nonpositive paired combined score across the training cells would disprove the mechanism."""
    n_nodes = embeddings.shape[0]
    n_dims = embeddings.shape[1]
    total_area = float(node_areas.sum())
    target_top = min(max(float(balance_target), 0.0), 1.0) * total_area

    seed_value = int(seed)
    first_dim = seed_value % max(1, n_dims)
    second_dim = (first_dim + 1) % max(1, n_dims)
    flip = 1.0 if (seed_value // max(1, n_dims)) % 2 == 0 else -1.0
    seed_projection = flip * (embeddings[:, first_dim] + 0.5 * embeddings[:, second_dim])

    anchor_top = int(np.argmax(seed_projection))
    anchor_bottom = int(np.argmin(seed_projection))
    centers = np.stack((embeddings[anchor_top], embeddings[anchor_bottom]), axis=0)
    labels = (np.arange(n_nodes) < max(1, n_nodes // 2)).astype(np.int64)

    for step_index in range(max_iterations):
        delta_top = embeddings - centers[0]
        delta_bottom = embeddings - centers[1]
        dist_top = np.sum(delta_top * delta_top, axis=1)
        dist_bottom = np.sum(delta_bottom * delta_bottom, axis=1)
        margin = dist_top - dist_bottom

        order = np.argsort(margin)
        cumulative_area = np.cumsum(node_areas[order])
        cut_pos = int(np.argmin(np.abs(cumulative_area - target_top)))
        cut_pos = min(max(cut_pos, 0), n_nodes - 2)

        rank_in_order = np.argsort(order)
        labels = (rank_in_order <= cut_pos).astype(np.int64)

        top_mask = labels == 1
        bottom_mask = labels == 0
        top_weight = np.where(top_mask, node_areas, 0.0)
        bottom_weight = np.where(bottom_mask, node_areas, 0.0)
        center_top = (embeddings * top_weight[:, None]).sum(axis=0) / max(float(top_weight.sum()), 1e-12)
        center_bottom = (embeddings * bottom_weight[:, None]).sum(axis=0) / max(float(bottom_weight.sum()), 1e-12)
        centers = np.stack((center_top, center_bottom), axis=0)

    return labels
