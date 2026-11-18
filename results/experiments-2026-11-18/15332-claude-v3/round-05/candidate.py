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
    """Unchanged from the parent: the maintained random-walk negative-sampling
    objective, `-(sum logsigmoid(pos) + sum logsigmoid(-neg)) / source_batch_size`.
    Kept identical here because this trial changes only the clustering decoder.
    """
    return -(
        F.logsigmoid(positive_scores).sum()
        + F.logsigmoid(-negative_scores).sum()
    ) / max(1, source_batch_size)


def decode_tp_gnn_embeddings(embeddings, node_areas, seed, max_iterations, balance_target):
    """Replace the parent's single reference_global_two_way handoff with one bounded
    custom two-center loop whose assignment step is a global area-balanced sweep.

    Change vs parent: instead of per-node nearest-center labeling plus whole-group
    rounding, each iteration ranks nodes by the squared-distance margin to the two
    area-weighted centers, cumulatively sums physical area along that ranking, and cuts
    where cumulative area first reaches `balance_target * total_area`. Centers are then
    refreshed from the resulting labels. Labels are oriented so the group whose
    area-weighted mean embedding projects higher on the area-weighted global mean
    embedding sits on the top die.

    Expected measurable effect: the returned labels spend the area balance directly, so
    the protected closure has almost nothing to repair and the discrete split differs
    from the parent's nearest-center partition mainly in the boundary margin band,
    changing final cut / timing-weighted cut and therefore TNS.

    Invalidation: if the labels stay identical (up to a die swap) to the parent on most
    cells, or the paired combined score does not exceed the parent's, this
    balanced-sweep mechanism is not what controls the physical timing.
    """
    emb = embeddings
    areas = node_areas
    total_area = float(areas.sum())
    safe_total = max(total_area, 1e-12)
    target_area = float(balance_target) * safe_total
    weights = areas / safe_total
    mean_vec = (emb * weights[:, None]).sum(axis=0)

    proj = (emb * mean_vec).sum(axis=1)
    order0 = np.argsort(proj)
    cum0 = np.cumsum(areas[order0])
    count = len(areas)
    k0 = int(np.argmin(np.abs(cum0 - target_area)))
    k0 = max(1, min(k0, count - 1))
    rank0 = np.argsort(order0)
    labels = (rank0 >= k0).astype(np.int64)

    for step in range(max_iterations):
        on_top = (labels == 0).astype(np.float64)
        on_bot = 1.0 - on_top
        w_top = areas * on_top
        w_bot = areas * on_bot
        area_top = max(float(w_top.sum()), 1e-12)
        area_bot = max(float(w_bot.sum()), 1e-12)
        center_top = (emb * w_top[:, None]).sum(axis=0) / area_top
        center_bot = (emb * w_bot[:, None]).sum(axis=0) / area_bot
        dist_top = ((emb - center_top) ** 2).sum(axis=1)
        dist_bot = ((emb - center_bot) ** 2).sum(axis=1)
        margin = dist_top - dist_bot
        order = np.argsort(margin)
        cum = np.cumsum(areas[order])
        cut = int(np.argmin(np.abs(cum - target_area)))
        cut = max(1, min(cut, count - 1))
        rank = np.argsort(order)
        labels = (rank >= cut).astype(np.int64)

    top_mask = (labels == 0).astype(np.float64)
    bot_mask = (labels == 1).astype(np.float64)
    a_top = max(float((areas * top_mask).sum()), 1e-12)
    a_bot = max(float((areas * bot_mask).sum()), 1e-12)
    avg_top = (emb * (areas * top_mask)[:, None]).sum(axis=0) / a_top
    avg_bot = (emb * (areas * bot_mask)[:, None]).sum(axis=0) / a_bot
    proj_top = float((avg_top * mean_vec).sum())
    proj_bot = float((avg_bot * mean_vec).sum())
    labels = np.where(proj_bot < proj_top, 1 - labels, labels)
    return labels
