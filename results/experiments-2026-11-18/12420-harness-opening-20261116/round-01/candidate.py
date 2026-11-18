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
    """Parent optimized only random-walk positive/negative pairs.

    Change relative to parent: add a normalized timing-weighted graph-star
    attraction term over the supplied represented-net edges. Since
    graph_scores are embedding dot products, (1 - graph_scores) is half the
    squared Euclidean distance between L2-normalized endpoints. Weighting it
    by graph_weights biases the embedding geometry toward keeping endpoints
    of high timing-weight nets close, so the area-balanced two-means decoder
    is more likely to place them on the same die.

    Expected measurable effect: lower timing-weighted cross-die cut and
    improved WNS/TNS versus the parent, especially on seeds where the parent
    cuts high-weight nets.

    Invalidating result: unchanged final labels/cut or worse mean timing
    across cases and seeds would reject this graph-attraction hypothesis.
    """
    base_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    weight_sum = torch.clamp(graph_weights.sum(), 1e-6)
    graph_attraction = (
        graph_weights * (1.0 - graph_scores)
    ).sum() / weight_sum

    pair_ratio = float(len(positive_scores) + len(negative_scores)) / max(
        1, source_batch_size
    )
    graph_scale = min(4.0, max(1.0, pair_ratio))

    return base_loss + 0.5 * graph_scale * graph_attraction


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Unchanged from parent.

    This loss_only candidate tests only the loss-induced embedding geometry.
    The decoder remains the frozen seeded area-balanced weighted two-way
    clustering helper, so any timing change must come from the loss rather
    than decoder behavior.
    """
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
