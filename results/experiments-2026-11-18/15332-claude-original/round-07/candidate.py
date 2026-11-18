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
    diversify edit substitutes a timing-feature-gated edge attraction for the
    uniform softplus edge mean. It reads graph_sources, graph_targets,
    node_features and graph_weights: each star edge is reweighted by clamped
    endpoint worst-slack criticality and timing coverage, and the focused term
    uses a squared margin. Because the attention factor can multiply
    timing-covered critical edge weights by up to about five and the squared
    margin has a steeper gradient than softplus near zero, the update is strong
    enough to change the embedding order at the area-balanced two-means
    boundary rather than only shrinking the loss. Expected measurable effect:
    timing-heavy connected groups stay together while unknown-timing edges are
    downweighted, changing the decoder's discrete top/bottom split and lowering
    timing-weighted cut/WNS/TNS. Invalidated if final layer assignments remain
    the parent's or timing-weighted cut does not fall.
    """
    pair_term = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    edge_count = max(1.0, float(len(graph_scores)))
    source_slack = node_features[graph_sources, 0]
    target_slack = node_features[graph_targets, 0]
    worst_slack = torch.where(source_slack < target_slack, source_slack, target_slack)
    source_coverage = node_features[graph_sources, 6]
    target_coverage = node_features[graph_targets, 6]
    coverage = torch.clamp(0.5 * (source_coverage + target_coverage), min=0.0, max=1.0)
    criticality = torch.clamp(-worst_slack, min=0.0, max=2.0)
    criticality_mean = criticality.sum() / edge_count
    criticality_ratio = criticality / (criticality_mean + 1e-3)
    attention = 1.0 + 3.0 * torch.clamp(criticality_ratio, min=0.0, max=2.0) * coverage
    edge_weights = graph_weights * attention
    base_edge = (graph_weights * F.softplus(-graph_scores)).sum() / (graph_weights.sum() + 1e-6)
    critical_edge = (edge_weights * torch.square(1.0 - graph_scores)).sum() / (edge_weights.sum() + 1e-6)
    pair_count = float(len(positive_scores)) + float(len(negative_scores))
    batch_scale = pair_count / max(1.0, float(source_batch_size))
    return pair_term + batch_scale * (base_edge + 3.0 * critical_edge)


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
