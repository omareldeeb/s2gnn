
import torch


def compute_triplets(edge_index: torch.Tensor, num_nodes: int = None):
    """
    Derive GemNet's triplet index tensors from a *directed* edge_index
    (source=row[0] = c, target=row[1] = a).

    Returns
    -------
    id3_expand_ba : (nTriplets,)  – edge-indices of b → a
    id3_reduce_ca : (nTriplets,)  – edge-indices of c → a  (the “central” edge)
    id_swap       : (nEdges,)     – maps every edge i (c→a) to the
                                    opposite edge j (a→c);  id_swap[i] = j
    Kidx3         : (nTriplets,)  – 0, 1, …, (#triplets for this c→a)-1,
                                    concatenated for every central edge
    """
    # ------------------------------------------------------------------ #
    # 1.  basic book-keeping
    # ------------------------------------------------------------------ #
    src, dst = edge_index  # src=c , dst=a
    M = src.size(0)
    device = src.device

    if num_nodes is None:
        num_nodes = int(torch.max(dst).item() + 1)

    # ------------------------------------------------------------------ #
    # 2.  gather incoming-edge lists for every potential target-atom a
    # ------------------------------------------------------------------ #
    # `edge_ids_per_target[a]` will hold all edge indices *ending* in a
    edge_ids_per_target = [[] for _ in range(num_nodes)]
    for e_idx, a in enumerate(dst.tolist()):
        edge_ids_per_target[a].append(e_idx)

    # ------------------------------------------------------------------ #
    # 3.  enumerate triplets   c→a   +   b→a,  with  b ≠ c
    #     • id3_reduce_ca : repeats the “central” edge  c→a
    #     • id3_expand_ba : the companion edge         b→a
    #     • Kidx3         : local index 0 … (#triplets_for_this_ca −1)
    # ------------------------------------------------------------------ #
    id3_reduce, id3_expand, kidx = [], [], []
    for edges_a in edge_ids_per_target:
        k = len(edges_a)
        if k < 2:                          # need at least two neighbours
            continue

        # cartesian product *without* the diagonal (b ≠ c)
        edges_a_tensor = torch.tensor(edges_a, device=device)

        # repeat_* builds an ordered cartesian product
        c_edges = edges_a_tensor.repeat_interleave(k)      # c index
        b_edges = edges_a_tensor.repeat(k)                 # b index
        mask = c_edges != b_edges                          # remove diagonal

        c_edges, b_edges = c_edges[mask], b_edges[mask]    # (k*(k-1),)

        id3_reduce.append(c_edges)
        id3_expand.append(b_edges)

        # local 0,… for this specific c→a edge
        # we get (#triplets) = k-1 for every *unique* c→a
        local_count = torch.arange(k - 1, device=device)
        kidx.append(local_count.repeat(k))  # one block per *central* edge

    id3_reduce_ca = torch.cat(id3_reduce, dim=0) if id3_reduce else torch.empty(0, dtype=torch.long, device=device)
    id3_expand_ba = torch.cat(id3_expand, dim=0) if id3_expand else torch.empty(0, dtype=torch.long, device=device)
    Kidx3         = torch.cat(kidx,        dim=0) if kidx        else torch.empty(0, dtype=torch.long, device=device)

    # ------------------------------------------------------------------ #
    # 4.  build  id_swap  (edge ↔ reverse-edge)
    # ------------------------------------------------------------------ #
    id_swap = torch.arange(M, device=device)  # default: self
    # dictionary mapping undirected pair → [forward_idx, reverse_idx]
    pair_dict = {}
    for e, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
        key = (v, u)     # look for *reverse* first
        if key in pair_dict:
            other = pair_dict[key]
            id_swap[e]      = other
            id_swap[other]  = e
        else:
            pair_dict[(u, v)] = e

    return id3_expand_ba, id3_reduce_ca, id_swap, Kidx3