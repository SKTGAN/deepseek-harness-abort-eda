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
    """Relative to the parent loss, keep the timing-weighted star-edge
    attraction sum_e w_e * softplus(-graph_scores_e) but focus each edge by the
    endpoint timing criticality read from published node_features: coverage-
    masked worst slack is standardized within the batch, and an exponential
    endpoint-importance multiplier makes edges incident to poor-slack groups
    dominate the graph term while damping edges whose timing is unknown. The
    multiplier spans roughly a factor of twelve, so it can reorder groups at
    the decoder's area-balanced boundary. Expected measurable effect: fewer
    TNS regressions on the large bp-ng45 cells while preserving the ariane133
    gains from the parent's timing-weighted attraction. Invalidated if the
    final layer assignment stays essentially unchanged, or if bp-ng45 TNS does
    not improve and the ariane133 gain is lost.
    """
    pair_term = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    raw_slack = node_features[:, 0]
    raw_coverage = node_features[:, 6]
    slack_values = torch.where(
        torch.isfinite(raw_slack), raw_slack, torch.zeros_like(raw_slack)
    )
    coverage = torch.clamp(raw_coverage, 0.0, 1.0)
    coverage = torch.where(
        torch.isfinite(coverage), coverage, torch.zeros_like(coverage)
    )
    coverage_total = coverage.sum() + 1e-6
    slack_mean = (coverage * slack_values).sum() / coverage_total
    slack_var = (
        coverage * torch.square(slack_values - slack_mean)
    ).sum() / coverage_total
    slack_scale = torch.sqrt(slack_var) + 1e-6
    slack_score = (slack_values - slack_mean) / slack_scale
    node_importance = coverage * torch.sigmoid(-slack_score)

    edge_importance = 0.5 * (
        node_importance[graph_sources] + node_importance[graph_targets]
    )
    focus_multiplier = torch.exp(2.5 * edge_importance - 1.25)
    focused_weights = graph_weights * focus_multiplier
    focused_total = focused_weights.sum()
    edge_mean = (
        focused_weights * F.softplus(-graph_scores)
    ).sum() / (focused_total + 1e-6)

    pair_count = float(len(positive_scores)) + float(len(negative_scores))
    batch_scale = pair_count / max(1.0, float(source_batch_size))
    return pair_term + batch_scale * edge_mean


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
