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
    """Add a stronger timing-weighted star-edge attraction term to the parent loss.

    Change relative to the parent: keep the sampled random-walk log-sigmoid
    contrastive objective unchanged, and add a normalized timing-weighted
    margin attraction on the supplied timing-context star edges.  The margin
    term pushes endpoints of timing-weighted connected nets toward a positive
    embedding dot product, with normalization by total graph edge weight so the
    graph term remains comparable across nets.

    Expected measurable effect: the timing-weighted graph attraction reshapes
    embeddings more strongly, so the unchanged area-balanced two-center decoder
    should move at least a few supernodes across the partition boundary and
    improve TNS where timing-weighted connectivity is informative.  The main
    risk is a WNS regression on bp_fe-ng45.

    Invalidation: if the official labels remain unchanged on most training
    cells, or if the combined score does not exceed the parent, this stronger
    graph-attraction mechanism is not useful.
    """
    contrastive_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    graph_margin_penalty = (
        graph_weights * F.softplus(0.5 - graph_scores)
    ).sum() / graph_weights.sum().clamp(min=1.0)

    return contrastive_loss + 4.0 * graph_margin_penalty


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Keep the parent frozen area-balanced global two-way decoder.

    No change relative to the parent: this still delegates exactly once to the
    trusted reference helper with unchanged arguments.  The hypothesis is
    isolated to the loss geometry, so the decoder's measurable behavior should
    remain unchanged.  If decoder-side behavior changes, the comparison does
    not test the intended loss-only hypothesis.
    """
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
