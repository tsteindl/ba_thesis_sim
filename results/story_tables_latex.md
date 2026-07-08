# Paste-ready LaTeX for the extra-info tables

## Robustness across thresholds (paper-relevant ranges only)

```latex
\begin{table}[ht]
\centering
\caption{Best adaptive-vs-baseline budget ratio across convergence thresholds (higher is better). Winner is the algorithm achieving the highest ratio.}
\label{tab:robustness-across-thresholds}
\begin{tabular}{lccccl}
\toprule
Scenario & 50\% & 80\% & 90\% & 95\% & Winner \\
\midrule
$\mathcal{U}(0.01,0.1)$, $\epsilon=$1e-03  & 1.35\texttimes & 1.27\texttimes & 1.23\texttimes & 1.12\texttimes & Reverse engineering \\
$\mathcal{U}(0.01,0.1)$, $\epsilon=$1e-04  & 1.84\texttimes & 1.70\texttimes & 1.56\texttimes & 1.47\texttimes & Reverse engineering \\
$\mathcal{U}(0.01,0.1)$, $\epsilon=$1e-05  & 2.13\texttimes & 1.87\texttimes & 1.72\texttimes & 1.58\texttimes & Reverse engineering \\
$\mathcal{U}(0.01,0.1)$, $\epsilon=$1e-06  & 2.16\texttimes & 1.88\texttimes & 1.72\texttimes & 1.61\texttimes & Reverse engineering \\
$\mathcal{U}(0.01,0.1)$, $\epsilon=$1e-07  & 2.17\texttimes & 1.88\texttimes & 1.72\texttimes & 1.61\texttimes & Reverse engineering \\
$\mathcal{U}(0.01,0.1)$, $\epsilon=$1e-08  & 2.17\texttimes & 1.88\texttimes & 1.72\texttimes & 1.61\texttimes & Reverse engineering \\
$\mathcal{U}(0.001,0.01)$, $\epsilon=$1e-03  & 0.97\texttimes & 1.09\texttimes & 1.02\texttimes & 0.97\texttimes & Linear search \\
$\mathcal{U}(0.001,0.01)$, $\epsilon=$1e-04  & 1.25\texttimes & 1.26\texttimes & 1.21\texttimes & 1.19\texttimes & Reverse engineering \\
$\mathcal{U}(0.001,0.01)$, $\epsilon=$1e-05  & 1.82\texttimes & 1.67\texttimes & 1.59\texttimes & 1.46\texttimes & Reverse engineering \\
$\mathcal{U}(0.001,0.01)$, $\epsilon=$1e-06  & 2.07\texttimes & 1.81\texttimes & 1.73\texttimes & 1.58\texttimes & Reverse engineering \\
$\mathcal{U}(0.001,0.01)$, $\epsilon=$1e-07  & 2.07\texttimes & 1.81\texttimes & 1.72\texttimes & 1.58\texttimes & Reverse engineering \\
$\mathcal{U}(0.001,0.01)$, $\epsilon=$1e-08  & 2.06\texttimes & 1.79\texttimes & 1.72\texttimes & 1.58\texttimes & Reverse engineering \\
$\mathcal{U}(0.001,0.1)$, $\epsilon=$1e-04  & 2.05\texttimes & 1.68\texttimes & 1.55\texttimes & 1.41\texttimes & Reverse engineering \\
$\mathcal{U}(0.001,0.1)$, $\epsilon=$1e-06  & 2.56\texttimes & 2.00\texttimes & 1.73\texttimes & 1.59\texttimes & Reverse engineering \\
\bottomrule
\end{tabular}
\end{table}
```

## Winning parameters at the 90% budget

```latex
\begin{table}[ht]
\centering
\caption{Winning parameters at the 90\% convergence budget.}
\label{tab:opt-param-second-tab}
\begin{tabular}{lllll}
\toprule
Algorithm
& $\epsilon=10^{-3}$, U(0.01,0.1)
& $\epsilon=10^{-4}$, U(0.01,0.1)
& $\epsilon=10^{-4}$, U(0.001,0.01)
& $\epsilon=10^{-4}$, U(0.001,0.1) \\
\midrule

L-search
& \makecell{\texttt{m\_exploration=28,} \\ \texttt{lookback\_window=2,} \\ \texttt{safeguard=3,} \\ \texttt{inc=2} \\ (@budget 43,218)}
& \makecell{\texttt{m\_exploration=97,} \\ \texttt{lookback\_window=5,} \\ \texttt{safeguard=1,} \\ \texttt{inc=1} \\ (@budget 3,135,633)}
& \makecell{\texttt{m\_exploration=3,} \\ \texttt{lookback\_window=5,} \\ \texttt{safeguard=3,} \\ \texttt{inc=2} \\ (@budget 416,087)}
& \makecell{\texttt{m\_exploration=74,} \\ \texttt{lookback\_window=5,} \\ \texttt{safeguard=2,} \\ \texttt{inc=1} \\ (@budget 3,451,483)} \\

\midrule

B-search
& \makecell{\texttt{m\_exploration=37,} \\ \texttt{safeguard=2,} \\ \texttt{conf=0.5} \\ (@budget 58,396)}
& \makecell{\texttt{m\_exploration=936,} \\ \texttt{safeguard=2,} \\ \texttt{conf=0.9} \\ (@budget 3,436,485)}
& \makecell{\texttt{m\_exploration=63,} \\ \texttt{safeguard=2,} \\ \texttt{conf=0.5} \\ (@budget 800,957)}
& \makecell{\texttt{m\_exploration=201,} \\ \texttt{safeguard=2,} \\ \texttt{conf=0.5} \\ (@budget 4,941,085)} \\

\midrule

Rev Eng
& \makecell{\texttt{m\_exploration=354,} \\ \texttt{safeguard=0.85} \\ (@budget 37,197)}
& \makecell{\texttt{m\_exploration=2691,} \\ \texttt{safeguard=0.95} \\ (@budget 2,890,389)}
& \makecell{\texttt{m\_exploration=192,} \\ \texttt{safeguard=0.85} \\ (@budget 362,887)}
& \makecell{\texttt{m\_exploration=8343,} \\ \texttt{safeguard=0.9} \\ (@budget 2,911,267)} \\

\bottomrule

\end{tabular}
\end{table}
```
