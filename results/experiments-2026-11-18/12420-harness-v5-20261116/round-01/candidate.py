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
    """Parent random-walk objective plus a timing-weighted star-edge
    consistency term on the supplied graph dot products. Each star edge is
    weighted by its frozen per-edge timing share (weights of one net sum to
    its full net weight), so nets with a large delay/slack spread dominate
    the attraction and their endpoint groups are pulled into the same
    embedding region, which the unchanged two-way decoder then keeps on one
    die. Expected measurable effect: lower timing-weighted cut and lower
    final WNS/TNS violation deficit at identical area-closure behaviour.
    The term is a weight-normalized average (finite and zero when the graph
    has no edges) and is ramped in with epoch progress so early epochs keep
    the parent scoring geometry."""
    base = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    safe_scores = torch.where(
        torch.isfinite(graph_scores), graph_scores, torch.zeros_like(graph_scores)
    )
    safe_weights = torch.where(
        torch.isfinite(graph_weights), graph_weights, torch.zeros_like(graph_weights)
    )
    total_weight = torch.clamp(torch.sum(safe_weights), min=1.0e-6)
    timing_attraction = torch.sum(
        safe_weights * (-F.logsigmoid(safe_scores))
    ) / total_weight
    warm = min(1.0, float(epoch + 1) / float(max(1, max_epochs)))
    return base + (1.5 * warm) * timing_attraction


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Unchanged parent decoder: the frozen area-balanced global two-way
    clustering, so any timing delta is attributable to the loss geometry."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
