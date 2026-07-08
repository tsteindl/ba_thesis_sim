# Broad-distribution LaTeX table

```latex
\begin{table}[ht]
\centering
\caption{Convergence under a broad uniform prior $\phi\sim\mathcal{U}(0.01,\pi/2)$ at $\epsilon=10^{-3}$: average share of simulations that converge, by budget. Linear search dominates at every budget while reverse engineering plateaus near $38\%$.}
\label{tab:broad-pi2}
\begin{tabular}{rcccc}
\toprule
Budget $C=N\cdot m$ & Brute force & Linear search & Binary search & Reverse engineering \\
\midrule
15{,}243 & 19.8\% & \textbf{23.0\%} & 17.5\% & 19.0\% \\
34{,}362 & 28.9\% & \textbf{34.7\%} & 27.4\% & 24.6\% \\
77{,}459 & 42.0\% & \textbf{47.4\%} & 40.9\% & 29.7\% \\
174{,}608 & 59.3\% & \textbf{65.8\%} & 59.0\% & 34.4\% \\
393{,}597 & 79.0\% & \textbf{83.2\%} & 77.4\% & 37.3\% \\
887{,}240 & 94.0\% & \textbf{95.1\%} & 92.0\% & 38.1\% \\
\bottomrule
\end{tabular}
\end{table}
```
