# Appendix code listings

A compact, executable implementation of the algorithms exactly as the thesis presents them
(Listings A.1–A.6). Tracing, tuning, compatibility variants and report generation are deliberately
omitted; the only dependencies are NumPy and `scipy.special.ndtr`. Use these files directly with
`\lstinputlisting` rather than filtering the production package.

The statistical safeguard is its own listing because both Binary Search and Reverse Engineering call
it.

```bash
python thesis_code/verify.py   # 200 fixed seeds per algorithm, vs qmetrology/algorithms.py
```

`verify.py` is a guard against drift, not a listing: it checks each compact function trial-for-trial
against the measured implementation on the headline scenario.

**Two pseudocode details still to reconcile in the .tex:**

1. Linear Search takes an `increment` parameter in the experiments, but the pseudocode signature
   omits it and writes `N <- N + 1`.
2. The pseudocode initialises the previous running mean to infinity, which counts the first probe as
   a decrease. The measured implementation initialises it to zero, so only decreases between two
   actual estimates count.
