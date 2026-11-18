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
    """Mechanism: retain timing-coverage and normalized slack-severity positive-pair weighting, but halve its amplitude to limit over-attraction from the most severe endpoint groups; Decision: high-severity connected groups still receive stronger embedding attraction, yet the maximum extra positive weight is reduced relative to the parent loss; Prediction: the bp_fe-ng45 seed 1001 regression narrows while the bp-ng45 and ariane133 gains largely persist, and timing-weighted cut remains below the unweighted baseline; Falsifier: a smaller mean paired combined-score delta than the parent across the training cells, or no reduction in the largest bp_fe-ng45 seed 1001 regression, would disprove the hypothesis."""
    coverage = torch.clamp(node_features[:, 6], min=0.0, max=1.0)
    slack_col = node_features[:, 0]
    slack_center = slack_col.mean()
    slack_scale = 1.0 + torch.sqrt(
        torch.mean(torch.square(slack_col - slack_center)) + 1e-6
    )
    severity = torch.sigmoid(-(slack_col - slack_center) / slack_scale)
    node_importance = coverage * severity
    positive_importance = 0.5 * (
        node_importance[positive_sources] + node_importance[positive_targets]
    )
    positive_weight = 1.0 + 0.5 * positive_importance
    positive_term = -(positive_weight * F.logsigmoid(positive_scores)).sum()
    negative_term = -F.logsigmoid(-negative_scores).sum()
    return (positive_term + negative_term) / max(1, source_batch_size)


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
