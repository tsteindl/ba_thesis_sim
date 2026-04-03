import numpy as np
from sim import simulate_errors


def find_phi_fixed_budget_binary_search(rng, phi, phi_max, phi_min, m_exploration, budget, lookback_window=5, safeguard=1):
    N_min = max(int(np.pi // (2 * phi_max)), 1)
    N_max = max(int(np.pi // (2 * phi_min)), 1)

    phi_hat_list = []   # rolling estimates
    N_history = []      # N used at each step
    budget_used = 0

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #
    def probe(N):
        """Run one exploration block, record results, return phi_hat."""
        nonlocal budget_used
        phi_hat = simulate_errors(rng, phi, m_exploration, N)
        budget_used += N * m_exploration
        phi_hat_list.append(phi_hat)
        N_history.append(N)
        return phi_hat

    def overshot():
        """
        True when the rolling mean has been strictly increasing for
        `lookback_window` consecutive steps (estimator degrading).
        Mirrors the linear search: we watch r_mean; if it went UP
        lookback_window times in a row the last N was too large.
        """
        if len(phi_hat_list) < lookback_window + 1:
            return False
        means = [np.mean(phi_hat_list[:k+1]) for k in range(len(phi_hat_list))]
        recent = means[-(lookback_window + 1):]          # last lw+1 values
        return all(recent[i] > recent[i-1] for i in range(1, len(recent)))

    def best_N_so_far():
        """
        Rewind by lookback_window steps to get the N that was optimal
        before the delayed criterion fired.
        """
        if len(N_history) > lookback_window:
            return N_history[-(lookback_window + 1)]
        return N_history[0]

    # ------------------------------------------------------------------ #
    # phase 1 – exponential expansion to find upper bracket               #
    # ------------------------------------------------------------------ #
    N = N_min
    N_lo = N_min
    N_hi = None

    probe(N)

    while budget_used < budget:
        N_next = min(N * 2, N_max)
        probe(N_next)

        if overshot():
            N_hi = N_next
            N_lo = best_N_so_far()        # rewind to pre-overshoot state
            break

        if N_next >= N_max:
            # reached hard ceiling without overshooting → use N_max
            N_lo = N_next
            N_hi = N_next
            break

        N = N_next

    # ------------------------------------------------------------------ #
    # phase 2 – binary search inside [N_lo, N_hi]                        #
    # ------------------------------------------------------------------ #
    while N_hi is not None and N_lo < N_hi and budget_used < budget:
        N_mid = (N_lo + N_hi) // 2
        if N_mid == N_lo:          # interval has collapsed
            break

        probe(N_mid)

        if overshot():
            N_hi = N_mid
            N_lo = max(N_lo, best_N_so_far())
        else:
            N_lo = N_mid

    # ------------------------------------------------------------------ #
    # phase 3 – exploit remaining budget at optimal N                     #
    # ------------------------------------------------------------------ #
    N_opt = max(1, N_lo - safeguard)

    remaining = budget - budget_used
    if remaining <= 0:
        return phi_hat_list[-1], budget_used

    m = int(remaining / N_opt)
    if m > 0:
        phi_hat_final = simulate_errors(rng, phi, m, N_opt)
        budget_used += m * N_opt
    else:
        phi_hat_final = phi_hat_list[-1]

    return phi_hat_final, budget_used