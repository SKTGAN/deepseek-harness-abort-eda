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
    """Diversify vs parent: retain the maintained random-walk contrastive term
    (positives attracted, sampled negatives pushed apart) and add one timing-
    weighted star-edge cohesion term over the supplied ``graph_scores`` and
    ``graph_weights``. The extra term nudges supernodes that share a weighted
    timing net toward each other in the L2 embedding, so it acts directly on
    the geometry the two-means decoder later partitions. The term is linearly
    scaled by a slow warmup so early epochs keep the parent objective. Expected
    measurable effect: a modestly different discrete die assignment versus the
    parent, with unchanged decoder call. Invalidation: if the final named
    assignment and cut metrics are identical across all seeds, the cohesion
    term did not move the partition. Only linear-in-E reductions are built, so
    memory stays bounded and no dense pairwise arrays are formed."""
    max_ep = max(1, int(max_epochs))
    warm = float(epoch) / float(max_ep)
    pos_term = F.logsigmoid(positive_scores).sum()
    neg_term = F.logsigmoid(-negative_scores).sum()
    walk_loss = -(pos_term + neg_term) / max(1, int(source_batch_size))
    edge_weight = graph_weights.sum()
    edge_anchor = (graph_weights * F.logsigmoid(graph_scores)).sum()
    cohesion = edge_anchor / (edge_weight + 1e-6)
    graph_loss = -0.5 * warm * cohesion
    return walk_loss + graph_loss


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Unchanged vs parent: delegate to the frozen area-balanced weighted
    two-means reference so the protected feasibility closure and handoff
    remain identical; only the loss geometry above is diversified."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
