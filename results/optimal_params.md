# Appendix — optimal parameters for the reported results

_Every adaptive number is the de-biased winner of a grid search (tuned seed 42, validated seed 2024). The full per-budget winners are in `results/story_winners.csv`; the tables below extract the ones behind the headline cells._

## Table 3.1 — winning parameters at fixed budget 10,000  (ε=10⁻³, U(0.01,0.1))

| Algorithm | optimal parameters |
|---|---|
| Linear search | `m_exploration=10, lookback_window=1, safeguard=5, inc=5` |
| Binary search | `m_exploration=11, safeguard=5, conf=0.5` |
| Reverse Engineering | `m_exploration=95, safeguard=0.8` |

## Table 3.2 — winning parameters at the 90%-convergence budget

| Algorithm | ε=10⁻³, U(0.01,0.1) | ε=10⁻⁴, U(0.01,0.1) | ε=10⁻⁴, U(0.001,0.01) | ε=10⁻⁴, U(0.001,0.1) |
|---|---|---|---|---|
| Linear search | `m_exploration=28, lookback_window=2, safeguard=3, inc=2` (@budget 46,001) | `m_exploration=97, lookback_window=5, safeguard=1, inc=1` (@budget 3,161,498) | `m_exploration=3, lookback_window=5, safeguard=3, inc=2` (@budget 439,502) | `m_exploration=74, lookback_window=5, safeguard=2, inc=1` (@budget 3,161,498) |
| Binary search | `m_exploration=37, safeguard=2, conf=0.5` (@budget 55,489) | `m_exploration=936, safeguard=2, conf=0.9` (@budget 3,161,498) | `m_exploration=63, safeguard=2, conf=0.5` (@budget 771,393) | `m_exploration=201, safeguard=2, conf=0.5` (@budget 4,600,122) |
| Reverse Engineering | `m_exploration=354, safeguard=0.85` (@budget 38,135) | `m_exploration=2691, safeguard=0.95` (@budget 2,620,925) | `m_exploration=192, safeguard=0.85` (@budget 364,353) | `m_exploration=8343, safeguard=0.9` (@budget 3,161,498) |

_Note the recurring pattern: the winning linear/RE configs use a **small exploration count** and commit the rest of the budget to exploitation at the inferred depth._
