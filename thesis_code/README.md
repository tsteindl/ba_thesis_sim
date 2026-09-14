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

Both listings track the pseudocode of `BA_Steindl.pdf`: Algorithm 4 carries `inc` and `w` in its
signature, initialises the window mean to zero and clamps the increment with `N <- min(N + inc,
N_max)`; Algorithm 7 takes a single pilot batch, without the `phi_hat_0 = 0` retry loop.
