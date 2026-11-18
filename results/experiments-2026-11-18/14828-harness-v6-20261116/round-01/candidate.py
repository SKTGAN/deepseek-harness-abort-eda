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
    """Unchanged maintained negative-sampling objective.

    This is a decoder-only experiment, so the embedding loss keeps its exact
    parent behaviour (plain random-walk positive/negative logistic loss). The
    expected measurable effect is zero change attributable to the loss; any
    WNS/TNS movement is attributable to the handoff split below.
    """
    return -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Balanced Lloyd split of the learned embeddings (replaces reference k-means).

    The reference helper optimises unconstrained area-weighted two-means and
    leaves the topology balance to the protected closure. Here the split
    direction is the difference of physical-area-weighted group centres and
    every step re-thresholds at the area-weighted quantile of the projected
    embeddings, so the returned labels already sit on the 0.5 area target and
    use exactly the area-weighted centres the protected squared-distance
    margin recomputes. Expected effect: fewer native closure moves and a
    lower timing-weighted cut, i.e. less negative WNS/TNS versus the parent.
    """
    n = int(embeddings.shape[0])
    area = np.abs(node_areas) + 1e-12
    w = area / area.sum()
    mu = (embeddings * w[:, None]).sum(axis=0)
    X = embeddings - mu
    var = (X * X * w[:, None]).sum(axis=0)
    axis = int(np.argmax(var))
    proj = X[:, axis]
    order = np.argsort(proj)
    cw = np.cumsum(w[order])
    k = int(np.sum(cw < balance_target))
    k = max(1, min(k, n - 1))
    lo = float(cw[k - 1])
    hi = float(cw[min(k, n - 1)])
    k = k + int(abs(hi - balance_target) < abs(lo - balance_target))
    k = max(1, min(k, n - 1))
    labels = (np.argsort(order) >= k) * 1
    w0 = w * (1 - labels)
    w1 = w * labels
    c0 = (X * (w0 / (w0.sum() + 1e-12))[:, None]).sum(axis=0)
    c1 = (X * (w1 / (w1.sum() + 1e-12))[:, None]).sum(axis=0)
    direction = c1 - c0
    for it in range(max_iterations):
        u = direction / (np.sqrt((direction * direction).sum()) + 1e-12)
        proj = (X * u[None, :]).sum(axis=1)
        order = np.argsort(proj)
        cw = np.cumsum(w[order])
        k = int(np.sum(cw < balance_target))
        k = max(1, min(k, n - 1))
        lo = float(cw[k - 1])
        hi = float(cw[min(k, n - 1)])
        k = k + int(abs(hi - balance_target) < abs(lo - balance_target))
        k = max(1, min(k, n - 1))
        labels = (np.argsort(order) >= k) * 1
        w0 = w * (1 - labels)
        w1 = w * labels
        c0 = (X * (w0 / (w0.sum() + 1e-12))[:, None]).sum(axis=0)
        c1 = (X * (w1 / (w1.sum() + 1e-12))[:, None]).sum(axis=0)
        direction = c1 - c0
    return labels
