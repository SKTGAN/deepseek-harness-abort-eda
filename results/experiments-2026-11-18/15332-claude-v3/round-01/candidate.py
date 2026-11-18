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
    """Add a normalized timing-weighted star-edge cut penalty to the parent loss.

    Change vs parent: the random-walk contrastive term is kept, and a
    timing-weighted ``softplus(-graph_scores)`` term is added over the
    published loss-context stars. This encourages endpoints of high-weight
    timing nets to receive same-side embeddings, which the unchanged decoder
    can then turn into fewer cut timing edges. Expected effect: lower
    timing-weighted cut and improved TNS, with WNS roughly neutral. The
    hypothesis is invalidated if paired seeds keep the same legal assignment
    and timing, or if timing regresses broadly across cases and seeds.
    """
    contrastive = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    weighted_cut = (
        graph_weights * F.softplus(-graph_scores)
    ).sum()
    weight_total = graph_weights.sum()
    star_penalty = weighted_cut / (weight_total + 1e-12)

    return contrastive + star_penalty


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
