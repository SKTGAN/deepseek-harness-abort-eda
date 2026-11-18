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
    """Parent logistic pair loss plus a timing-weighted star-edge cut proxy.

    Change relative to the parent: keep the random-walk logistic objective
    unchanged and add a small normalized penalty
    sum(graph_weights * sigmoid(graph_scores)) / sum(abs(graph_weights)).
    This uses the supplied timing-context stars and pushes apart embeddings
    on high-weight timing nets.

    Expected measurable effect: the learned embedding geometry changes enough
    to alter the area-balanced two-means partition, potentially reducing TNS
    through a lower timing-weighted cut, while WNS may trade off.

    Invalidating result: labels and final timing remain essentially matched
    to the parent across seeds/cases, or the added term causes broad timing
    regression rather than a reproducible combined-score gain.
    """
    random_walk_loss = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    graph_weight_mass = torch.clamp(
        torch.sum(torch.abs(graph_weights)), min=1e-12
    )
    timing_star_penalty = torch.sum(
        graph_weights * torch.sigmoid(graph_scores)
    ) / graph_weight_mass

    return random_walk_loss + 0.05 * timing_star_penalty


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Unchanged relative to the parent: use the frozen area-balanced decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
