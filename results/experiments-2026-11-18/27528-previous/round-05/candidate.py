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
    """Parent random-walk logistic pair objective; change vs parent is an added
    timing-weighted graph-star alignment term on supplied star-edge dot products.
    Expected measurable effect: timing-critical connected supernodes receive more
    similar embeddings, so the same decoder can reduce timing-weighted cut and TNS.
    Invalidating result: unchanged layer assignments or no timing-score gain.
    """
    pair_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    graph_denom = torch.clamp(graph_weights.sum(), min=1e-6)
    graph_margin = torch.clamp(1.0 - graph_scores, min=0.0)
    graph_term = (
        graph_weights * torch.square(graph_margin)
    ).sum() / graph_denom

    graph_scale = 6.0
    return pair_loss + graph_scale * graph_term


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
