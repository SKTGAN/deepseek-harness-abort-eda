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
    """Mechanism: retain the random-walk contrastive loss and the timing-weighted graph-star attraction term, but renormalize graph edges by squared timing weights; Decision: emphasize high-weight timing-critical net-star edges in the embedding gradient while leaving the decoder and all non-graph loss terms unchanged, so the tier decision near decoder boundaries is driven more by timing-critical connections; Prediction: timing-weighted cut should fall and matched-case combined timing should improve across cases and seeds with smaller cross-seed regression than linear edge weighting; Falsifier: matched official combined score fails to improve while decoder assignments and timing-weighted cut remain unchanged."""
    pair_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    graph_weight_pos = torch.clamp(graph_weights, min=0.0)
    graph_weight_sq = torch.square(graph_weight_pos)
    graph_weight_total = torch.clamp(graph_weight_sq.sum(), min=1e-12)
    graph_star_loss = (
        graph_weight_sq * F.softplus(-graph_scores)
    ).sum() / graph_weight_total
    return pair_loss + 0.75 * graph_star_loss


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Mechanism: preserve the frozen seeded area-balanced two-means handoff and replace the parent's one-sided high-margin micro-flip with a two-sided balanced swap of groups that are closer to the opposite area-weighted center, using equal cumulative moved-area and count caps on each side; Decision: change a group's tier only when a wrong-side group on the opposite tier of comparable cumulative area is also moved, so the local swap does not shift the global area balance or center orientation; Prediction: the largest one-sided-drift regressions should shrink while the boundary gains persist, giving higher mean combined timing across seeds and a lower timing-weighted cut; Falsifier: matched official combined score fails to improve over the parent while the selected swaps are absent or the timing-weighted cut rises."""
    labels = reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
    areas = np.maximum(node_areas, 1e-12)
    total_area = np.sum(areas)
    top_mask = labels == 1
    bottom_mask = labels == 0
    top_area = np.sum(areas * top_mask)
    bottom_area = np.sum(areas * bottom_mask)
    top_center = np.sum(embeddings * (areas * top_mask)[:, None], axis=0) / np.maximum(top_area, 1e-12)
    bottom_center = np.sum(embeddings * (areas * bottom_mask)[:, None], axis=0) / np.maximum(bottom_area, 1e-12)
    d_top = np.sum((embeddings - top_center) ** 2, axis=1)
    d_bottom = np.sum((embeddings - bottom_center) ** 2, axis=1)
    margin = d_bottom - d_top
    violation = np.where(labels == 1, -margin, margin)
    top_candidate = top_mask & (violation > 0.0)
    bottom_candidate = bottom_mask & (violation > 0.0)
    top_priority = np.where(top_candidate, violation / areas, -1.0)
    bottom_priority = np.where(bottom_candidate, violation / areas, -1.0)
    top_order = np.argsort(-top_priority)
    bottom_order = np.argsort(-bottom_priority)
    top_sorted_candidate = top_candidate[top_order]
    bottom_sorted_candidate = bottom_candidate[bottom_order]
    top_sorted_areas = areas[top_order]
    bottom_sorted_areas = areas[bottom_order]
    top_cum_area = np.cumsum(np.where(top_sorted_candidate, top_sorted_areas, 0.0))
    bottom_cum_area = np.cumsum(np.where(bottom_sorted_candidate, bottom_sorted_areas, 0.0))
    top_cum_count = np.cumsum(top_sorted_candidate)
    bottom_cum_count = np.cumsum(bottom_sorted_candidate)
    max_side_area = 0.001 * total_area
    max_side_count = max(1, int(0.001 * labels.size))
    top_total_candidate_area = np.sum(areas * top_candidate)
    bottom_total_candidate_area = np.sum(areas * bottom_candidate)
    move_area_limit = min(max_side_area, top_total_candidate_area, bottom_total_candidate_area)
    top_selected_sorted = (
        top_sorted_candidate
        & (top_cum_area <= move_area_limit)
        & (top_cum_count <= max_side_count)
    )
    bottom_selected_sorted = (
        bottom_sorted_candidate
        & (bottom_cum_area <= move_area_limit)
        & (bottom_cum_count <= max_side_count)
    )
    top_selected = top_selected_sorted[np.argsort(top_order)]
    bottom_selected = bottom_selected_sorted[np.argsort(bottom_order)]
    new_labels = np.where(top_selected, 0, labels)
    new_labels = np.where(bottom_selected, 1, new_labels)
    top_count = np.sum(new_labels == 1)
    bottom_count = labels.size - top_count
    if top_count == 0 or bottom_count == 0:
        return labels
    return new_labels
