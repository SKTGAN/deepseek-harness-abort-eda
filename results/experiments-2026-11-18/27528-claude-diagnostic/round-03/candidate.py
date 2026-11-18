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
    """Change vs the published parent: retain the random-walk positive/negative
    objective and add a timing-weighted star-edge cut penalty that is gated by
    endpoint timing pressure from node_features. This is the slack-gated variant
    of the prior uniform timing-weighted star-edge penalty: only represented nets
    whose graph endpoints have negative worst slack and nonzero timing coverage
    receive cut pressure. Expected measurable effect: preserve WNS while improving
    TNS by pulling negative-slack connected groups toward the same decoder side.
    Invalidation: if the macro-heavy training case at seed 1002 does not improve
    TNS or if the smaller bp case regresses relative to the parent.
    """
    pair_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    coverage = torch.clamp(node_features[:, 6], min=0.0, max=1.0)
    worst_slack = node_features[:, 0]
    node_slack_pressure = torch.clamp(-worst_slack, min=0.0) * coverage
    edge_slack_pressure = (
        node_slack_pressure[graph_sources] + node_slack_pressure[graph_targets]
    )
    slack_gate = edge_slack_pressure / (edge_slack_pressure + 0.1)
    gated_weight = graph_weights * slack_gate
    graph_penalty = (
        gated_weight * (1.0 - graph_scores)
    ).sum() / (gated_weight.sum() + 1e-6)

    return pair_loss + 0.5 * graph_penalty


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Unchanged from the parent: use the frozen area-balanced global two-way
    decoder so the loss-only edit is isolated from clustering behavior.
    """
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
