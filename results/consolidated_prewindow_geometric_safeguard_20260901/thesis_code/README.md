# Proposed Appendix code

These files are a compact, executable implementation of the algorithms as presented in the thesis.
They intentionally omit tracing, tuning, compatibility variants and report-generation code.  They
have not been copied into `thesis/code/`; review them first, then use the reviewed files directly
with `\lstinputlisting` rather than generating filtered source from the production package.

`verify.py` checks the compact functions trial-for-trial against the measured implementations on
200 fixed headline-scenario seeds.  It is a guard against drift and is not intended as a listing.

The statistical safeguard is a separate listing because both Binary Search and Reverse Engineering
call it in the pseudocode.  The only external dependencies are NumPy and `scipy.special.ndtr`, which
implements the normal CDF used by the safeguard and the Binary Search threshold.

Before replacing the current Appendix listings, reconcile two pseudocode details:

1. Linear Search uses an `increment` parameter in the experiments, but its pseudocode signature
   currently omits it and writes `N <- N + 1`.
2. The pseudocode initializes the previous running mean to infinity.  That counts the first probe as
   a decrease; the measured implementation initializes it to zero so that only decreases between
   two actual estimates count.
3. RESOLVED.  The safeguard pseudocode, the compact listing and the measured implementation now
   describe one rule: score every integer of `N_min,...,N_max` and take the argmax, with the
   accuracy factor evaluated at the whole shot count `m = floor(B/N)`.  The earlier divergence --
   pseudocode with `floor(B/N)` over `N_min,...,N_max` against a measured
   `2 Phi(2 epsilon sqrt(N B))-1` found by three rounds of geometric refinement from a lower bound
   of 1 -- is gone, and the reported numbers were regenerated against the new rule.
