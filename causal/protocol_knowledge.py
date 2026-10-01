"""Protocol knowledge as a constraint ON discovery, not a post-hoc relabelling.

The baseline pipeline (causal.discovery) runs PC unconstrained and applies
protocol precedence afterwards, so precedence can only re-orient an edge that
already survived PC. This module supplies the same knowledge to the search
itself, plus the two structural facts the search would otherwise be free to
violate.

Three kinds of knowledge, all stated before any data is seen:

  1. LAYER ORDER (forbidden direction only). A later protocol layer cannot
     cause an earlier one within a scheduling slot: L3 -/-> L2 -/-> L1, and no
     endogenous variable causes an exogenous one.

  2. WITHIN-LAYER PROCESSING ORDER (forbidden direction only). A receiver
     measures SINR before it selects an MCS, and decodes before it computes
     spectral efficiency. This forbids the backward direction; it asserts NO
     adjacency. Nodes of equal rank are left mutually unconstrained, because
     protocol order genuinely does not sequence them.

  3. EXOGENOUS MUTUAL INDEPENDENCE (both directions). Two exogenous variables
     never cause one another. This is definitional rather than empirical --
     it is what makes the SCM Markovian -- and it is the only constraint here
     that removes candidate ADJACENCIES: causal-learn drops an edge from
     background knowledge only when both directions are forbidden.

None of this references the ground-truth edge list. Ordering is available from
the 3GPP processing chain before any measurement; adjacency is the thing under
test.
"""
import numpy as np

from .discovery import NODE_LAYER

# Protocol processing order INSIDE each layer. rank(a) < rank(b) forbids
# b -> a and says nothing about whether a -> b exists. Equal rank = no
# constraint either way.
WITHIN_RANK = {
    # L1 PHY: measure SINR -> select MCS -> demodulate/decode -> compute SE
    "sinr_eff": 0, "mcs_idx": 1,
    "mod_order": 2, "code_rate": 2, "bler": 2,
    "se_phy": 3,
    # L2 MAC: grant -> retransmission accounting -> realised throughput
    "num_prb": 0,
    "harq_retx": 1,
    "mac_tput": 2,
    # L3 network: queue state -> loss / latency -> delivered goodput
    "queue_dep": 0,
    "pkt_loss": 1, "rtt_ms": 1,
    "goodput": 2,
}


def protocol_position(name):
    """Total order used for 'earlier than': layer, then within-layer rank."""
    return (NODE_LAYER[name], WITHIN_RANK.get(name, 0))


def is_exogenous_pair(a, b):
    return NODE_LAYER[a] == 0 and NODE_LAYER[b] == 0


def violates_protocol_order(edge):
    """True if the edge runs backwards through the within-layer processing order."""
    a, b = edge
    if NODE_LAYER[a] != NODE_LAYER[b]:
        return False
    ra, rb = WITHIN_RANK.get(a), WITHIN_RANK.get(b)
    return ra is not None and rb is not None and ra > rb


def forbidden(edge):
    """The structural constraints, in the one place both stages consult."""
    return violates_protocol_order(edge) or is_exogenous_pair(*edge)


def prohibited_edges(columns):
    """ALL three kinds of knowledge above, as an explicit set of edges.

    forbidden() covers kinds 2 and 3 only: the pipeline delivers kind 1 (layer
    order) to PC through causal-learn tiers in build_background_knowledge, so
    forbidden() never needed it. A learner that takes an edge list instead --
    the BLINC-style baseline -- would silently get LESS knowledge than PC from
    forbidden() alone. This is the complete set, and a test pins it to exactly
    what build_background_knowledge forbids.
    """
    return {(a, b) for a in columns for b in columns
            if a != b and (NODE_LAYER[a] > NODE_LAYER[b] or forbidden((a, b)))}


def build_background_knowledge(columns):
    """causal-learn BackgroundKnowledge carrying all three kinds above."""
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    nodes = {c: GraphNode(c) for c in columns}
    bk = BackgroundKnowledge()
    for c in columns:                                   # 1. layer tiers
        bk.add_node_to_tier(nodes[c], NODE_LAYER[c])
    for a in columns:
        for b in columns:
            if a == b:
                continue
            if NODE_LAYER[a] == NODE_LAYER[b]:          # 2. within-layer order
                ra, rb = WITHIN_RANK.get(a), WITHIN_RANK.get(b)
                if ra is not None and rb is not None and ra < rb:
                    bk.add_forbidden_by_node(nodes[b], nodes[a])
            if is_exogenous_pair(a, b):                 # 3. exogenous, both ways
                bk.add_forbidden_by_node(nodes[a], nodes[b])
    return bk


def run_pc_constrained(obs, columns, alpha=0.01, bk=None):
    """PC with background knowledge -> (directed, undirected) on column names.

    NOTE: causal-learn's pc() limits conditioning-set size with `max_k`, not
    `depth`. causal.discovery.run_pc passes `depth=`, which is not a pc()
    parameter and is silently discarded, so that path searches at unlimited
    depth. This function does not pass either, and therefore also searches at
    unlimited depth -- deliberately, and stated rather than implied. Measured
    difference between max_k=2 and unlimited on this problem: F1 0.610 vs
    0.605.
    """
    from causallearn.search.ConstraintBased.PC import pc
    cg = pc(obs[columns].values.astype(float), alpha=alpha,
            indep_test="fisherz", stable=True, show_progress=False,
            node_names=list(columns), background_knowledge=bk)
    g = cg.G.graph
    directed, undirected = [], []
    for i in range(len(columns)):
        for j in range(len(columns)):
            if g[i, j] == -1 and g[j, i] == 1:
                directed.append((columns[i], columns[j]))
            elif i < j and g[i, j] == -1 and g[j, i] == -1:
                undirected.append((columns[i], columns[j]))
    return directed, undirected


def _reaches(adj, src, dst):
    seen, stack = set(), [src]
    while stack:
        n = stack.pop()
        if n == dst:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, ()))
    return False


def is_acyclic(edges):
    nodes = {n for e in edges for n in e}
    indeg = {n: 0 for n in nodes}
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        indeg[b] += 1
    q = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while q:
        n = q.pop()
        seen += 1
        for m in adj.get(n, ()):
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    return seen == len(nodes)


def acyclic_projection(base, candidates):
    """Admit candidate edges into an acyclic base, most-significant-first.

    The target is an SCM over a DAG, so a cyclic output is not a debatable
    modelling choice -- it is outside the hypothesis class. PC + precedence +
    DirectLiNGAM is already acyclic; the augmentation stage is what can close a
    cycle, because it only checks LAYER order and so admits both directions
    between two equal-rank nodes in the same layer.

    base       : iterable of edges, assumed acyclic
    candidates : {edge: p_value}; admitted in increasing p-value order, and
                 skipped when the edge would close a cycle.
    """
    out, adj = set(base), {}
    for a, b in out:
        adj.setdefault(a, []).append(b)
    for (a, b) in sorted(candidates, key=lambda e: candidates[e]):
        if _reaches(adj, b, a):
            continue
        out.add((a, b))
        adj.setdefault(a, []).append(b)
    return out


def enforce(edges):
    """Re-apply every structural constraint to a finished edge set."""
    kept, adj = set(), {}
    for a, b in sorted(e for e in edges if not forbidden(e)):
        if _reaches(adj, b, a):
            continue
        kept.add((a, b))
        adj.setdefault(a, []).append(b)
    return kept
