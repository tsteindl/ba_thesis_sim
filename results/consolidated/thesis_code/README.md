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
3. The safeguard pseudocode computes the accuracy factor with the integer
   `m=floor(B/N)` and scans `N_min,...,N_max`.  The measured implementation uses
   `2 Phi(2 epsilon sqrt(N B))-1`, a three-round geometric refinement followed by a local integer
   scan, and permits Reverse Engineering to consider values below `N_min`.  The compact listing
   follows the measured implementation so that it reproduces the reported results.
