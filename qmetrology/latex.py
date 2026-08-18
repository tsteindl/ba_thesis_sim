"""LaTeX emitters for every table in the thesis.

The style is fixed by the existing manuscript and reproduced exactly: `booktabs` rules, `\\makecell`
for stacked column headers, `{,}` thousands separators (so LaTeX does not add spurious math spacing),
`\\texttimes` for the improvement factors, and `A. \\ref{alg:...}` row labels. Nothing here reads a
CSV or runs a simulation — the builders take numbers in and return a string — so the same functions
serve analysis/make_tex.py (which renders from the result CSVs) and any ad-hoc use.

Requires in the preamble: \\usepackage{booktabs} and \\usepackage{makecell}.
"""
import numpy as np

# row identity, shared by every table so the labels never drift apart between them
REF = {
    "brute": r"A. \ref{alg:brute-force}: Brute force baseline",
    "linear": r"A. \ref{alg:linear-search}: Linear search",
    "binary": r"A. \ref{alg:binary-search}: Binary search",
    "reverse_eng": r"A. \ref{alg:reverse-engineering}: Reverse Engineering",
    "binary_risk": r"A. \ref{alg:binary-search}: Binary search (Eq.~\eqref{eq:risk-depth})",
    "reverse_eng_risk": r"A. \ref{alg:reverse-engineering}: Reverse Eng. (Eq.~\eqref{eq:risk-depth})",
    "separable": r"Separable protocol ($N=1$)",
    "oracle_hl": r"\textit{Ceiling} (best depth, Eq.~\eqref{eq:risk-depth} law)",
    "oracle": r"\textit{Oracle} ($N=N_{\mathrm{opt}}$, knows $\phi$)",
    "oracle_alias": r"\textit{Oracle, no safeguard} ($N=\lfloor\pi/2\phi\rfloor$)",
}
# short forms for tables whose first column is narrow
SHORT = {"linear": "L-search", "binary_risk": "B-search", "reverse_eng_risk": "Rev Eng",
         "binary": r"B-search (tuned $s$)", "reverse_eng": r"Rev Eng (tuned $C$)",
         "brute": "Brute force", "separable": "Separable", "oracle_hl": "Ceiling"}
# the statistical safeguard replaces the tuned constant, so the *_risk keys carry the plain names
PLAIN = {"brute": "Brute force", "linear": "Linear search", "separable": "Separable ($N=1$)",
         "binary_risk": "Binary search", "reverse_eng_risk": "Reverse engineering",
         "binary": "Binary search (tuned $s$)", "reverse_eng": "Reverse engineering (tuned $C$)",
         "oracle_hl": "Attainable ceiling"}
MU = {"U(0.01,0.1)": r"\mathcal{U}(0.01,0.1)", "U(0.001,0.01)": r"\mathcal{U}(0.001,0.01)",
      "U(0.001,0.1)": r"\mathcal{U}(0.001,0.1)", "U(0.0001,0.1)": r"\mathcal{U}(0.0001,0.1)",
      "U(0.0001,0.01)": r"\mathcal{U}(0.0001,0.01)", "U(0.005,0.05)": r"\mathcal{U}(0.005,0.05)",
      "U(0.0001,0.001)": r"\mathcal{U}(0.0001,0.001)", "U(1e-05,0.01)": r"\mathcal{U}(10^{-5},0.01)"}
DASH = "--"


# ---------------------------------------------------------------- cell formatters
def num(x, dashes=DASH):
    """42{,}195 — the braces stop LaTeX typesetting the comma as a math separator."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dashes
    return f"{x:,.0f}".replace(",", "{,}")


def pct(x, dec=2, dashes=DASH):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dashes
    return f"{x:.{dec}f}\\%"


def factor(x, dec=2, dashes=DASH):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dashes
    return f"\\texttimes {x:.{dec}f}"


def eps_math(eps):
    """1e-04 (str or float) -> $10^{-4}$."""
    e = int(round(np.log10(float(eps))))
    return f"$10^{{{e}}}$"


def ci(lo, hi, fmt=num, dashes=""):
    r"""Compact 95% interval, e.g. '{[45{,}179, 46{,}646]}'.

    The outer braces are load-bearing: a cell starting with `[` directly after a `\\` is parsed by
    LaTeX as the optional vertical-space argument of `\\`, which is exactly what happens when an
    interval is stacked under a value inside \makecell.
    """
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return dashes
    return f"{{[{fmt(lo)}, {fmt(hi)}]}}"


def pm(x, lo, hi, fmt=num, dashes=DASH):
    """Value with a symmetric-looking half-width, e.g. '45{,}868 $\\pm$ 734'."""
    if x is None or not np.isfinite(x):
        return dashes
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return fmt(x)
    return f"{fmt(x)} $\\pm$ {fmt((hi - lo) / 2)}"


def stack(*lines):
    r"""\makecell{a \\ b \\ c} — the manuscript's stacked-cell idiom."""
    return r"\makecell{" + r" \\ ".join(str(x) for x in lines if x != "") + "}"


def texttt(**params):
    r"""\texttt{m\_exploration=10, safeguard=5} with the underscores escaped."""
    body = ", ".join(f"{k}={v}" for k, v in params.items()).replace("_", r"\_")
    return f"\\texttt{{{body}}}"


def bold(s):
    return f"\\textbf{{{s}}}"


# ---------------------------------------------------------------- table skeleton
def table(colspec, header, rows, caption, label, *, size=None, placement="ht",
          note=None, blank_between_rows=False, midrules=()):
    """Assemble one booktabs table.

    colspec   : the tabular column specification, e.g. "lcc"
    header    : list of header cells (joined with &) or a pre-joined string
    rows      : list of rows; each row is a list of cells or a pre-joined string
    size      : e.g. "scriptsize" / "small" to shrink a wide table
    note      : text appended under the tabular inside the table environment
    midrules  : row indices (0-based) to precede with \\midrule
    """
    L = [f"\\begin{{table}}[{placement}]", r"\centering"]
    if size:
        L.append(f"\\{size}")
    L += [f"\\caption{{{caption}}}", f"\\label{{{label}}}",
          f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    L.append((header if isinstance(header, str) else "\n& ".join(header)) + r" \\")
    L.append(r"\midrule")
    for i, r in enumerate(rows):
        if i in midrules:
            L.append(r"\midrule")
        L.append((r if isinstance(r, str) else "\n& ".join(r)) + r" \\")
        if blank_between_rows:
            L.append("")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L.append(f"\\\\[2pt]\n\\footnotesize {note}")
    L.append(r"\end{table}")
    return "\n".join(L)
