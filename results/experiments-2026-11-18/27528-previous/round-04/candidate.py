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
    """Margin-hinge bifurcation objective instead of the parent soft logistic.

    Change vs the parent initial program: the parent's logistic pair objective
    is replaced by a hard hinge margin (0.5) on the sampled positive and
    negative pair scores, and the timing-weighted net-star scores enter as an
    additional hinge constraint normalised by their fixed total weight with a
    mild epoch ramp.  Logistic terms saturate once a pair is already well
    separated, so the learned cloud stayed nearly unimodal and the seeded
    area-balanced two-means partition barely moved from the initial one.  A
    margin hinge keeps a constant gradient on unsaturated hard pairs and on
    the heaviest timing nets, which should sharpen the two-lobe embedding
    geometry that the decoder's two-means ranking can actually exploit.

    Expected measurable effect: a materially larger partition change versus
    the initial assignment, a lower timing-weighted cut, and better TNS
    deficit improvement on the timing-dominated bp-ng45 cell.  The hypothesis
    is invalidated if the partition stays mostly identical to the initial one,
    or if the timing-weighted cut rises while TNS deficit improvement does not
    improve.
    """
    margin = 0.5
    batch = max(1, source_batch_size)
    positive_term = F.relu(margin - positive_scores).sum()
    negative_term = F.relu(margin + negative_scores).sum()
    pair_term = (positive_term + negative_term) / batch
    star_weight_total = torch.clamp(graph_weights.sum(), min=1.0e-06)
    star_violation = (graph_weights * F.relu(margin - graph_scores)).sum()
    star_term = star_violation / star_weight_total
    progress = float(epoch) / float(max(1, max_epochs))
    star_coeff = 1.0 + 2.0 * progress
    return pair_term + star_coeff * star_term


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
