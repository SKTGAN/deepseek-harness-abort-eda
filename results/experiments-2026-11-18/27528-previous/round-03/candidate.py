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
    """Parent random-walk logistic pair objective plus timing-weighted graph-star attraction.
    Change vs parent: add a normalized -logsigmoid(graph_scores) term weighted by graph_weights.
    Expected measurable effect: endpoints of timing-critical nets become more similar, reducing
    timing-weighted cut and improving WNS/TNS. Invalidated if final timing or partition assignment
    does not improve, or if pair separation collapses.
    """
    pair_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    weight_total = graph_weights.sum()
    safe_total = torch.clamp(weight_total, min=1e-12)
    edge_term = -(F.logsigmoid(graph_scores) * graph_weights).sum() / safe_total
    graph_penalty = torch.where(weight_total > 0.0, edge_term, torch.zeros_like(edge_term))
    graph_coeff = 0.5
    return pair_loss + graph_coeff * graph_penalty


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
