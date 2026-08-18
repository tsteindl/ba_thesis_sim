# Appendix — optimal parameters for the reported results

_Every adaptive number is the de-biased winner of a grid search (tuned seed 42, validated seed 2024). The full per-budget winners are in `results/story_winners.csv`; the tables below extract the ones behind the headline cells._

## Table 3.1 — winning parameters at fixed budget 10,000  (ε=10⁻³, U(0.01,0.1))

| Algorithm | optimal parameters | pilot share of budget |
|---|---|---|
| Linear search | `m_exploration=10, lookback_window=1, safeguard=1, inc=5` | — |
| Binary search | `m_exploration=109, conf=0.5, eps_target=0.001` | — |
| Reverse Engineering | `m_exploration=71, eps_target=0.001` | 10.7% |

_Reverse engineering's exploration size is quoted in the thesis as a **share of the budget**, ρ = 2% (floor 20 shots), not as a shot count: a fixed m′ can only be correct at one budget. Over the 546 operating points of `analysis/re_share_sweep.py` the fixed ρ costs −0.06 pp against tuning m′ at every budget separately, while a fixed m′ = 200 costs −5.55 pp (worst −99.4 pp)._

## Table 3.2 — winning parameters at the 90%-convergence budget

| Algorithm | ε=10⁻³, U(0.01,0.1) | ε=10⁻⁴, U(0.01,0.1) | ε=10⁻⁴, U(0.001,0.01) | ε=10⁻⁴, U(0.001,0.1) |
|---|---|---|---|---|
| Linear search | `m_exploration=14, lookback_window=2, safeguard=2, inc=2` (@budget 46,001) | `m_exploration=97, lookback_window=5, safeguard=1, inc=1` (@budget 3,161,498) | `m_exploration=3, lookback_window=5, safeguard=5, inc=2` (@budget 364,353) | `m_exploration=166, lookback_window=5, safeguard=2, inc=2` (@budget 3,161,498) |
| Binary search | `m_exploration=321, conf=0.5, eps_target=0.001` (@budget 31,614) | `m_exploration=201, conf=0.65, eps_target=0.0001` (@budget 3,161,498) | `m_exploration=433, conf=0.5, eps_target=0.0001` (@budget 364,353) | `m_exploration=6407, conf=0.5, eps_target=0.0001` (@budget 2,620,925) |
| Reverse Engineering | `m_exploration=67, eps_target=0.001` (@budget 31,614) — **3.2%** of budget | `m_exploration=2691, eps_target=0.0001` (@budget 2,620,925) — **1.5%** of budget | `m_exploration=61, eps_target=0.0001` (@budget 302,053) — **3.2%** of budget | `m_exploration=1266, eps_target=0.0001` (@budget 2,620,925) — **0.7%** of budget |

_Note the recurring pattern: the winning linear/RE configs use a **small exploration count** and commit the rest of the budget to exploitation at the inferred depth._

_Read as a fraction of the budget, reverse engineering's tuned exploration sizes above span **0.7–3.2%** — the same few percent at every budget and prior, which is why the thesis quotes ρ = 2% rather than a shot count. The corresponding m′ values span 20–300,000, a range of four orders of magnitude._
