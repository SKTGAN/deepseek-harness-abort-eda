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
    """Mechanism: retain the random-walk contrastive loss and add a timing-weighted graph-star attraction term over supplied graph edges; Decision: the embedding geometry is pulled toward positive L2 dot products on high-weight net edges, giving timing-critical connected groups a lower cut cost before the frozen area-balanced two-means decoder assigns tiers; Prediction: matched-case combined timing should improve where timing-weighted cuts are reduced, with labels changing near decoder decision boundaries rather than by case- or seed-specific rules; Falsifier: matched-case combined score fails to improve while timing-weighted cut and decoder assignments remain unchanged."""
    pair_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    graph_weight_total = torch.clamp(graph_weights.sum(), min=1e-6)
    graph_star_loss = (
        graph_weights * F.softplus(-graph_scores)
    ).sum() / graph_weight_total
    return pair_loss + 0.75 * graph_star_loss


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
