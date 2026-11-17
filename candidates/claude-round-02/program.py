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
    """Relative to the parent (random-walk contrastive loss only), add a
    timing-weighted star-edge attraction term
        sum_e w_e * softplus(-graph_scores_e) / sum_e w_e
    rescaled by (P+Q)/source_batch_size so its total gradient is comparable to
    the existing pair term and is strong enough to reorder groups at the
    area-balanced two-means boundary. Expected measurable effect: embeddings
    keep timing-heavy nets locally close, lowering the timing-weighted cut and
    shifting the official decoder's discrete top/bottom split, improving
    WNS/TNS. The hypothesis is invalidated if the final layer assignment is
    essentially unchanged or the timing-weighted cut does not fall.
    """
    pair_term = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    weight_total = graph_weights.sum()
    edge_mean = (graph_weights * F.softplus(-graph_scores)).sum() / (weight_total + 1e-6)
    pair_count = float(len(positive_scores)) + float(len(negative_scores))
    batch_scale = pair_count / max(1.0, float(source_batch_size))
    return pair_term + batch_scale * edge_mean


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
