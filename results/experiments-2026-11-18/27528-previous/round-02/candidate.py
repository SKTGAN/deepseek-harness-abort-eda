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
    """Parent random-walk pair objective with timing/area pair reweighting.

    Mechanism: keep the parent logistic positive/negative pair geometry, but
    weight each sampled pair by endpoint timing-criticality and physical area.
    Criticality uses timing coverage, bounded worst-slack pressure, and a
    clamped normalized area share, so the loss emphasizes well-covered,
    timing-suspect nodes without inferring macro identity from area.

    Hypothesis: upweighting pairs at timing-critical, well-covered nodes should
    move the two-means handoff split toward a timing-weighted cut and reduce
    TNS deficit on large cases relative to the parent, with the decoder
    unchanged.

    Invalidating result: if coverage is zero or slack saturates so weights are
    nearly uniform, or if the frozen two-means assignment is insensitive to the
    shifted embedding geometry, final labels and TNS deficit should remain
    unchanged or worsen.
    """
    coverage = torch.clamp(node_features[:, 6], 0.0, 1.0)
    slack_gate = 0.5 + 0.5 * torch.tanh(-node_features[:, 0])
    area_total = node_areas.sum()
    area_share = node_areas / (area_total + 1.0e-8)
    area_scale = torch.sqrt(
        torch.clamp(
            area_share * max(1, int(node_features.shape[0])),
            0.0,
            4.0,
        )
    )
    importance = 1.0 + 0.75 * coverage * slack_gate + 0.25 * area_scale
    positive_weight = 0.5 * (
        importance[positive_sources] + importance[positive_targets]
    )
    negative_weight = 0.5 * (
        importance[negative_sources] + importance[negative_targets]
    )
    positive_term = -(positive_weight * F.logsigmoid(positive_scores)).sum()
    negative_term = -(negative_weight * F.logsigmoid(-negative_scores)).sum()
    return (positive_term + negative_term) / max(1, source_batch_size)


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Frozen parent area-balanced global two-way decoder; unchanged."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
