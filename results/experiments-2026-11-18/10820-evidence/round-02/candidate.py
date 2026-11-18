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
    """Mechanism: add a timing-weighted margin on star-edge dot products to the maintained contrastive loss; Decision: high-weight star edges receive extra gradient so the unchanged two-means decoder tends to place timing-connected supernodes in one tier; Prediction: timing-weighted cut and combined timing improve versus the parent, especially on bp-ng45; Falsifier: no reduction in timing-weighted cut or nonpositive mean paired combined-score delta across seeds would disprove the hypothesis."""
    contrastive = -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)

    edge_margin = 0.2
    edge_cost = F.softplus(edge_margin - graph_scores)
    graph_denom = torch.clamp(graph_weights.sum(), min=1e-6)
    graph_consistency = (graph_weights * edge_cost).sum() / graph_denom
    return contrastive + 4.0 * graph_consistency


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Reproduce the frozen area-balanced global two-way decoder."""
    return reference_global_two_way(embeddings, node_areas, seed, max_iterations, balance_target)
