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
    """Relative to the parent timing-weighted star-edge attraction, this
    substitutes an area-mobility-weighted edge mean read from the published
    graph endpoints and node_areas. Small movable standard-cell supernodes
    receive stronger timing attraction, while large hard-to-move groups are
    damped, so the embedding geometry is less likely to be undone by the
    protected macro/area feasibility closure. The term stays a weighted edge
    mean rescaled by (P+Q)/source_batch_size, keeping its gradient comparable
    to the pair loss and strong enough to reorder groups at the area-balanced
    two-means boundary. Expected measurable effect: lower closure repair area,
    lower timing-weighted cut, and improved WNS/TNS on cases such as bp-ng45.
    Invalidated if the final layer assignment is unchanged, or if repair area
    and timing-weighted cut do not fall.
    """
    pair_term = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    mean_area = torch.clamp(node_areas.mean(), 1e-6)
    area_sum = node_areas[graph_sources] + node_areas[graph_targets]
    mobility = torch.clamp((2.0 * mean_area) / (area_sum + 1e-6), 0.1, 10.0)
    mobility_weight = graph_weights * mobility
    mobility_total = mobility_weight.sum()
    mobility_edge = (
        mobility_weight * F.softplus(-graph_scores)
    ).sum() / (mobility_total + 1e-6)

    pair_count = float(len(positive_scores)) + float(len(negative_scores))
    batch_scale = pair_count / max(1.0, float(source_batch_size))
    return pair_term + batch_scale * mobility_edge


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Relative to the parent, keep the timing-weighted star-edge loss
    unchanged and replace the frozen two-means helper with a custom seeded
    area-balanced two-means decoder. Initial centers are an embedding anchor
    and its farthest group; each bounded iteration re-ranks groups by the
    squared-distance margin and re-estimates physical-area-weighted centers.
    Because this directly decides the discrete top-die fill order, it is strong
    enough to change the partition. Expected measurable effect: a different
    legal split that improves measured WNS/TNS and timing-weighted cut.
    Invalidated if final layer assignments match the parent's or official
    timing does not improve.
    """
    n_nodes = int(embeddings.shape[0])
    total_area = node_areas.sum()
    mean_all = (node_areas[:, None] * embeddings).sum(axis=0) / max(float(total_area), 1e-12)

    anchor_index = int(seed) % max(1, n_nodes)
    anchor = embeddings[anchor_index]
    anchor_dist = ((embeddings - anchor) ** 2).sum(axis=1)
    far_index = int(np.argmax(anchor_dist))

    center_top = embeddings[far_index]
    center_bottom = embeddings[anchor_index]
    labels = np.where(np.arange(n_nodes) < max(1, n_nodes // 2), 1, 0)

    for iteration in range(max_iterations):
        dist_top = ((embeddings - center_top) ** 2).sum(axis=1)
        dist_bottom = ((embeddings - center_bottom) ** 2).sum(axis=1)
        score = dist_bottom - dist_top
        score_tie = score - (np.arange(n_nodes) * 1e-12)
        order = np.argsort(-score_tie)
        sorted_areas = node_areas[order]
        cumulative = np.cumsum(sorted_areas)
        target_area = balance_target * total_area
        errors = np.abs(cumulative - target_area)
        count = int(np.argmin(errors)) + 1
        count = max(1, min(n_nodes - 1, count))
        threshold = score_tie[order[count - 1]]
        labels = np.where(score_tie >= threshold, 1, 0)

        top_mask = labels == 1
        bottom_mask = labels == 0
        top_area = node_areas[top_mask].sum()
        bottom_area = node_areas[bottom_mask].sum()
        center_top = (
            node_areas[top_mask, None] * embeddings[top_mask]
        ).sum(axis=0) / max(float(top_area), 1e-12)
        center_bottom = (
            node_areas[bottom_mask, None] * embeddings[bottom_mask]
        ).sum(axis=0) / max(float(bottom_area), 1e-12)

    return labels
