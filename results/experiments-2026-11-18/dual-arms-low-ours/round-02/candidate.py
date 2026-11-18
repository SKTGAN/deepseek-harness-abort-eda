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
    """Change vs parent (loss_only): keep the maintained random-walk contrastive
    term and add one timing-weighted star-edge cohesion term computed from the
    supplied `graph_scores` and `graph_weights`. The frozen weight
    `1 + alpha*omega` (omega built from connection delay minus slack) is
    normalized into a probability-like edge mass, and `softplus(-dot)` is
    minimized so that net endpoints with larger timing weight are pulled closer
    in the same L2 metric the decoder clusters with.

    Expected measurable effect: the seeded, area-balanced weighted two-means
    sees embeddings whose geometry already respects timing-critical
    connectivity, so the discrete top/bottom partition should shift toward
    lower timing-weighted cut, improving paired WNS/TNS after closure and STA.

    Invalidation: if paired combined scores and WNS/TNS do not move, or the
    organizer partition diagnostic stays at the initial assignment with
    identical timing, the added term did not change the discrete decision and
    the hypothesis is rejected.
    """
    contrastive = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    weight_total = torch.clamp(graph_weights.sum(), 1e-6)
    edge_mass = graph_weights / weight_total
    graph_cohesion = (edge_mass * F.softplus(-graph_scores)).sum()
    return contrastive + graph_cohesion


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
