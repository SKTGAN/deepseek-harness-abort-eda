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
    """Hypothesis: add a timing-excess weighted star-edge attraction term so high-weight critical nets stay close in embedding space, reducing cross-die critical nets after the unchanged decoder."""
    base_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    edge_net_weights = net_weights[graph_net_ids]
    safe_net_weights = torch.where(
        torch.isfinite(edge_net_weights),
        edge_net_weights.clamp(min=1.0),
        torch.ones_like(edge_net_weights),
    )
    timing_excess = (
        graph_weights * (1.0 - 1.0 / safe_net_weights)
    ).clamp(min=0.0)

    score_finite = torch.where(
        torch.isfinite(graph_scores),
        torch.ones_like(graph_scores),
        torch.zeros_like(graph_scores),
    )
    critical_weights = torch.where(
        torch.isfinite(timing_excess),
        timing_excess,
        torch.zeros_like(timing_excess),
    ) * score_finite

    safe_graph_scores = torch.where(
        torch.isfinite(graph_scores),
        graph_scores,
        torch.zeros_like(graph_scores),
    )
    edge_denom = critical_weights.sum().clamp(min=1e-6)
    timing_loss = -(
        critical_weights * F.logsigmoid(safe_graph_scores)
    ).sum() / edge_denom

    return base_loss + 2.0 * timing_loss


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Keep the frozen area-balanced global two-way decoder from the selected parent unchanged."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
