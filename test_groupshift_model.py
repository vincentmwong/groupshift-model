import numpy as np
import pytest

from groupshift_model import AsymmetricSample


@pytest.mark.parametrize("sample_type", ["random", "extremity", "proximity"])
def test_sample_scope_is_indexed_by_group_id(sample_type):
    # GLEAN mechanics update the perception of group g from scope[i][g],
    # so scope[i][g] must hold members of group g whatever group node i is in.
    np.random.seed(0)
    n = 400
    N = np.random.uniform(0, 100, size=(n, 1))
    N_adj = np.zeros((n, 2))
    N_adj[n // 2:, 1] = 1
    G = np.zeros((n, 2, 1, 2))
    affected = np.array([0, 1, n - 2, n - 1])  # two nodes from each group

    scope = AsymmetricSample(50, 50, sample_type).apply(G, N, N_adj, affected, 1)

    for samples in scope:
        for g, members in enumerate(samples):
            assert len(members) == 50
            assert np.all(N_adj[members, 1] == g)
