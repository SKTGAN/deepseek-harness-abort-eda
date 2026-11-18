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
    """Hypothesis: timing-weighted full-net star smoothness, added to the random-walk contrastive objective, keeps high-criticality connected supernodes in one embedding neighborhood so the unchanged area-balanced decoder cuts fewer timing-critical nets."""
    pair_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    slack = 0.5 * node_features[:, 0] + 0.5 * node_features[:, 1]
    coverage = torch.clamp(node_features[:, 6], 0.0, 1.0)
    slack_mean = torch.mean(slack)
    slack_std = torch.sqrt(
        torch.mean(torch.square(slack - slack_mean)) + 1e-6
    )
    slack_z = (slack - slack_mean) / slack_std
    criticality = torch.sigmoid(-slack_z) * coverage

    edge_criticality = 0.5 * (
        criticality[graph_sources] + criticality[graph_targets]
    )
    edge_weights = graph_weights * (1.0 + edge_criticality)
    graph_loss = -(
        edge_weights * F.logsigmoid(graph_scores)
    ).sum() / (edge_weights.sum() + 1e-12)

    return pair_loss + 0.5 * graph_loss


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Unchanged maintained area-balanced global two-way clustering decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
