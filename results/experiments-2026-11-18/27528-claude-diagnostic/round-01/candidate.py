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
    """Change vs parent: retains the random-walk pair objective but adds a
    normalized timing-weighted star-edge cut penalty on graph_scores, using
    graph_weights so timing-critical represented nets are discouraged from
    crossing the two-way embedding boundary. Expected measurable effect:
    fewer high-weight cross-boundary star edges after the unchanged balanced
    decoder, giving a TNS-led timing gain with roughly neutral WNS. The
    hypothesis is invalidated if a paired evaluation shows no reproducible
    gain, mostly unchanged labels/timing, or broad regression versus parent.
    """
    safe_graph_scores = torch.where(
        torch.isfinite(graph_scores),
        graph_scores,
        torch.zeros_like(graph_scores),
    )
    safe_graph_weights = torch.where(
        torch.isfinite(graph_weights),
        graph_weights,
        torch.zeros_like(graph_weights),
    )
    graph_weight_total = safe_graph_weights.sum().clamp(min=1e-6)
    graph_cut_penalty = (
        safe_graph_weights * F.softplus(-4.0 * safe_graph_scores)
    ).sum() / graph_weight_total
    pair_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    return pair_loss + 0.5 * graph_cut_penalty


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Unchanged from parent: calls the frozen seeded area-balanced global
    two-way decoder so feasibility closure and handoff behavior are identical.
    """
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
