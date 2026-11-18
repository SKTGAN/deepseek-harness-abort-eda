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
    """Use a timing-weighted margin attraction on graph star edges.

    Change relative to the parent: keep the maintained random-walk
    positive/negative log-sigmoid contrastive term, and add a normalized
    timing-weighted graph penalty ``softplus(0.25 - graph_scores)``.  The
    positive margin asks critical connected groups not merely to avoid a
    negative dot product but to separate from the decision boundary, which
    should make the area-balanced two-way decoder choose a different discrete
    partition more readily than a plain zero-margin graph penalty.

    Expected measurable effect: lower timing-weighted cut and improved TNS,
    with a visible change in decoder labels or closure movement on at least a
    subset of cells while retaining the parent contrastive behavior.

    Hypothesis is invalidated if paired WNS/TNS and final labels remain at the
    parent levels, or if the larger graph attraction collapses the embedding
    split so that feasibility closure has to move many groups.
    """
    contrastive_term = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)
    graph_denominator = torch.clamp(graph_weights.sum(), 1e-6)
    graph_margin_penalty = (
        graph_weights * F.softplus(0.25 - graph_scores)
    ).sum() / graph_denominator
    return contrastive_term + 2.0 * graph_margin_penalty


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Keep the parent frozen area-balanced global two-way decoder unchanged.

    Change relative to the parent: none in the decoder.  This preserves the
    seeded, area-weighted two-means initialization and protected feasibility
    closure so that any measured timing difference is attributable to the new
    loss geometry rather than decoder mechanics.
    """
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
