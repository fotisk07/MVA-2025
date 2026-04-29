from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfnn_control import (
    ABMConfig,
    ABMSimulationResult,
    CentralityHeuristicPolicy,
    EncoderConfig,
    TrainingConfig,
    apply_initial_shock,
    build_global_bsde_networks,
    build_policy,
    core_periphery_graph_weights,
    default_mask,
    erdos_renyi_graph_weights,
    euler_step,
    expand_graph_weights,
    homogeneous_graph_weights,
    rollout_abm,
    sample_initial_states_with_dim,
)


# ---------------------------------------------------------------------------
# Bootstrap CI utilities
# ---------------------------------------------------------------------------

def bootstrap_ci(
    samples: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    stat: str = "mean",
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI.  stat='mean' | 'rate' (fraction > 0)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(samples, dtype=float)
    if len(arr) < 2:
        v = float(arr.mean()) if stat == "mean" else float((arr > 0).mean())
        return v, v
    if stat == "mean":
        boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    elif stat == "rate":
        boots = (rng.choice(arr, size=(n_boot, len(arr)), replace=True) > 0).mean(axis=1)
    else:
        raise ValueError(f"Unknown stat '{stat}'. Use 'mean' or 'rate'.")
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def add_cascade_ci(stats_dict: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Augment a cascade_stats dict in-place with 95% bootstrap CIs."""
    dist = stats_dict.get("cascade_size_distribution", [])
    if len(dist) < 2:
        stats_dict["cascade_rate_ci"] = [stats_dict["cascade_rate"]] * 2
        stats_dict["mean_cascade_size_ci"] = [stats_dict["mean_cascade_size"]] * 2
        return stats_dict
    rate_lo, rate_hi = bootstrap_ci(dist, n_boot=n_boot, stat="rate", seed=seed)
    size_lo, size_hi = bootstrap_ci(dist, n_boot=n_boot, stat="mean", seed=seed)
    stats_dict["cascade_rate_ci"] = [rate_lo, rate_hi]
    stats_dict["mean_cascade_size_ci"] = [size_lo, size_hi]
    return stats_dict


def fit_cascade_tail_gof(
    sizes: list[float],
    x_min: float = 1.0,
) -> dict[str, object]:
    """Fit exponential, log-normal, and Pareto (power-law) distributions to
    cascade sizes >= x_min and compare via AIC.

    Returns MLE parameters, log-likelihoods, AIC, and KS p-values for each
    candidate distribution, plus the AIC-preferred model name.
    """
    from scipy.stats import expon, lognorm, pareto, kstest

    data = np.asarray([s for s in sizes if s >= x_min], dtype=float)
    n = len(data)
    if n < 5:
        return {"n": n, "x_min": x_min, "best_model": "insufficient_data"}

    results: dict[str, object] = {"n": int(n), "x_min": float(x_min)}

    # Exponential
    shifted = data - x_min
    lam_hat = 1.0 / float(shifted.mean()) if shifted.mean() > 0 else 1e-9
    ll_exp = float(np.sum(np.log(lam_hat) - lam_hat * shifted))
    aic_exp = 2.0 * 1 - 2.0 * ll_exp
    ks_exp, ks_p_exp = kstest(shifted, lambda x: 1.0 - np.exp(-lam_hat * x))
    results["exponential"] = {
        "lambda": float(lam_hat),
        "log_likelihood": ll_exp,
        "aic": float(aic_exp),
        "ks_stat": float(ks_exp),
        "ks_p": float(ks_p_exp),
    }

    # Log-normal
    log_data = np.log(data)
    mu_ln = float(log_data.mean())
    sigma_ln = float(log_data.std(ddof=1)) if n > 1 else 1e-9
    if sigma_ln < 1e-9:
        sigma_ln = 1e-9
    ll_ln = float(np.sum(-np.log(data) - np.log(sigma_ln) - 0.5 * ((log_data - mu_ln) / sigma_ln) ** 2))
    aic_ln = 2.0 * 2 - 2.0 * ll_ln
    ks_ln, ks_p_ln = kstest(data, lambda x: lognorm.cdf(x, s=sigma_ln, scale=np.exp(mu_ln)))
    results["lognormal"] = {
        "mu": float(mu_ln),
        "sigma": float(sigma_ln),
        "log_likelihood": ll_ln,
        "aic": float(aic_ln),
        "ks_stat": float(ks_ln),
        "ks_p": float(ks_p_ln),
    }

    # Pareto / power-law
    alpha_pareto = float(n / np.sum(np.log(data / x_min)))
    ll_pareto = float(n * np.log(alpha_pareto) - n * np.log(x_min)
                      - (alpha_pareto + 1.0) * np.sum(np.log(data / x_min)))
    aic_pareto = 2.0 * 1 - 2.0 * ll_pareto
    ks_pa, ks_p_pa = kstest(data, lambda x: pareto.cdf(x, b=alpha_pareto, scale=x_min))
    results["pareto"] = {
        "alpha": float(alpha_pareto),
        "log_likelihood": ll_pareto,
        "aic": float(aic_pareto),
        "ks_stat": float(ks_pa),
        "ks_p": float(ks_p_pa),
    }

    # AIC-preferred model
    model_aics = {
        "exponential": float(aic_exp),
        "lognormal": float(aic_ln),
        "pareto": float(aic_pareto),
    }
    best = min(model_aics, key=lambda k: model_aics[k])
    delta = {k: float(v - model_aics[best]) for k, v in model_aics.items()}
    results["model_aic"] = model_aics
    results["delta_aic"] = delta
    results["best_model"] = best
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/systemic_risk_global_bsde_cylindrical_case_1_seed7.pt")
    parser.add_argument("--output-dir", default="results/abm")
    parser.add_argument("--case", default="case_1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state-dim", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--horizon", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--default-threshold", type=float, default=-0.5)
    parser.add_argument("--target-cascade-rate", type=float, default=0.35)
    parser.add_argument("--q-min", type=float, default=0.1)
    parser.add_argument("--q-max", type=float, default=1.4)
    parser.add_argument("--q-steps", type=int, default=14)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--core-hubs", type=int, default=10)
    parser.add_argument("--er-p", type=float, default=0.08)
    parser.add_argument("--mc-paths", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-grid", default="10,30,100,300,1000")
    parser.add_argument("--q-grid", default="0.4,0.6,0.8,1.0,1.2")
    parser.add_argument("--phase-q-min", type=float, default=0.0)
    parser.add_argument("--phase-q-max", type=float, default=2.0)
    parser.add_argument("--phase-q-steps", type=int, default=21)
    parser.add_argument("--subcrit-sigmas", default="0.2,0.5,1.0")
    parser.add_argument("--subcrit-q-min", type=float, default=0.2)
    parser.add_argument("--subcrit-q-max", type=float, default=2.0)
    parser.add_argument("--subcrit-q-steps", type=int, default=19)
    return parser.parse_args()


def _to_int_list(csv: str) -> list[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def _to_float_list(csv: str) -> list[float]:
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


def build_topologies(agents: int, core_hubs: int, er_p: float, device: str) -> dict[str, torch.Tensor]:
    return {
        "homogeneous": homogeneous_graph_weights(agents, device=device),
        "core_periphery": core_periphery_graph_weights(agents, hubs=min(core_hubs, max(1, agents - 1)), device=device),
        "erdos_renyi": erdos_renyi_graph_weights(agents, edge_probability=er_p, device=device),
    }


def highest_degree_node(weights: torch.Tensor) -> int:
    binary = (weights > 0).to(torch.int64)
    degrees = binary.sum(dim=1) - 1
    return int(torch.argmax(degrees).item())


class _BSDENetworkWrapper(torch.nn.Module):
    """Exposes a network(inputs) interface over a MeanFieldInitialValue network.

    inputs shape: (B, N, 1 + state_dim + feature_dim)  — same format produced by
    policy_actions_from_local_measures.  We ignore the pre-computed local features
    and instead let the initial-value network recompute global mean-field features
    from the raw states, then return the adjoint-induced control.
    """

    def __init__(self, initial_value_network: torch.nn.Module, q: float) -> None:
        super().__init__()
        self.initial_value_network = initial_value_network
        self.q = q
        self.state_dim: int = getattr(initial_value_network, "state_dim", 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        states = inputs[..., 1 : 1 + self.state_dim]
        adjoint = self.initial_value_network(states)
        mean_state = states.mean(dim=1, keepdim=True)
        return self.q * (mean_state - states) - adjoint


class _BSDEPolicyWrapper(torch.nn.Module):
    """Wraps BSDE networks to expose the encoder/network interface expected by
    policy_actions_from_local_measures and CentralityHeuristicPolicy.
    """

    def __init__(self, initial_value_network: torch.nn.Module, q: float) -> None:
        super().__init__()
        self.encoder = initial_value_network.encoder
        self.network = _BSDENetworkWrapper(initial_value_network, q)


def load_policy_from_checkpoint(checkpoint_path: Path, device: str) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder_config = EncoderConfig(**payload["encoder_config"])
    training_config = TrainingConfig(**payload["training_config"])
    if "policy_state_dict" in payload:
        policy = build_policy(encoder_config, training_config).to(device=device, dtype=torch.float32)
        policy.load_state_dict(payload["policy_state_dict"])
        policy.eval()
        return policy
    if "initial_value_state_dict" in payload:
        initial_value_network, _ = build_global_bsde_networks(encoder_config, training_config)
        initial_value_network = initial_value_network.to(device=device, dtype=torch.float32)
        initial_value_network.load_state_dict(payload["initial_value_state_dict"])
        initial_value_network.eval()
        q = float(payload["config"]["q"])
        return _BSDEPolicyWrapper(initial_value_network, q)
    raise ValueError("Checkpoint contains neither policy_state_dict nor initial_value_state_dict")


def default_times(states: torch.Tensor, threshold: float, initial_defaulted: torch.Tensor) -> torch.Tensor:
    hit = (states < threshold).any(dim=-1)
    hit = hit & (~initial_defaulted.unsqueeze(0))
    t_count, batch_size, agents = hit.shape
    out = torch.full((batch_size, agents), t_count - 1, device=states.device, dtype=torch.long)
    for t in range(t_count):
        new = hit[t] & (out == t_count - 1)
        out[new] = t
    never = (~hit).all(dim=0)
    out[never] = -1
    return out


def cascade_stats(result, threshold: float, initial_defaulted: torch.Tensor) -> dict[str, float | list[float]]:
    initial_counts = initial_defaulted.float().sum(dim=1)
    final_defaults = result.defaulted.float().sum(dim=1)
    additional_defaults = (final_defaults - initial_counts).clamp_min(0.0)
    cascade_rate = (additional_defaults > 0).float().mean().item()
    default_time = default_times(result.states, threshold, initial_defaulted)
    mean_first_default_time = default_time[default_time >= 0].float().mean().item() if (default_time >= 0).any() else -1.0
    return {
        "mean_cascade_size": float(additional_defaults.mean().item()),
        "cascade_rate": float(cascade_rate),
        "cascade_size_distribution": [float(x) for x in additional_defaults.cpu().tolist()],
        "mean_first_default_time": float(mean_first_default_time),
    }


def sample_initial(case: str, batch_size: int, agents: int, state_dim: int, device: str) -> torch.Tensor:
    return sample_initial_states_with_dim(case, batch_size, agents, state_dim, device=device, dtype=torch.float32)


def run_rollout(
    abm_config: ABMConfig,
    weights: torch.Tensor,
    case: str,
    mc_paths: int,
    state_dim: int,
    device: str,
    policy: torch.nn.Module | None,
) -> dict[str, object]:
    x0 = sample_initial(case, mc_paths, weights.shape[0], state_dim, device)
    node = highest_degree_node(weights)
    x0 = apply_initial_shock(x0, [node], abm_config.default_threshold - 0.2)
    initial_defaulted = (x0 < abm_config.default_threshold).any(dim=-1)
    sim = rollout_abm(x0, abm_config, weights, policy=policy)
    stats = cascade_stats(sim, abm_config.default_threshold, initial_defaulted)
    add_cascade_ci(stats)
    return {
        "hub_node": node,
        "stats": stats,
    }


def calibrate_q(
    base_config: ABMConfig,
    weights: torch.Tensor,
    case: str,
    mc_paths: int,
    state_dim: int,
    device: str,
    q_values: list[float],
    target_rate: float,
) -> dict[str, object]:
    sweep = []
    best_q = q_values[0]
    best_gap = 1e9
    for q in q_values:
        cfg = ABMConfig(
            horizon=base_config.horizon,
            steps=base_config.steps,
            sigma=base_config.sigma,
            interaction_q=q,
            state_dim=base_config.state_dim,
            default_threshold=base_config.default_threshold,
        )
        run = run_rollout(cfg, weights, case, mc_paths, state_dim, device, policy=None)
        rate = float(run["stats"]["cascade_rate"])
        gap = abs(rate - target_rate)
        if gap < best_gap:
            best_gap = gap
            best_q = q
        sweep.append({"q": float(q), "cascade_rate": rate})
    return {"selected_q": float(best_q), "target_cascade_rate": float(target_rate), "sweep": sweep}


def experiment_1_uncontrolled(
    abm_config: ABMConfig,
    topologies: dict[str, torch.Tensor],
    case: str,
    mc_paths: int,
    state_dim: int,
    device: str,
) -> dict[str, object]:
    out = {}
    for name, weights in topologies.items():
        out[name] = run_rollout(abm_config, weights, case, mc_paths, state_dim, device, policy=None)
    return out


def experiment_2_controlled(
    abm_config: ABMConfig,
    topologies: dict[str, torch.Tensor],
    case: str,
    mc_paths: int,
    state_dim: int,
    device: str,
    policy: torch.nn.Module,
) -> dict[str, object]:
    out = {}
    for name, weights in topologies.items():
        uncontrolled = run_rollout(abm_config, weights, case, mc_paths, state_dim, device, policy=None)
        controlled = run_rollout(abm_config, weights, case, mc_paths, state_dim, device, policy=policy)
        
        # Heuristic baseline: scale control by degree centrality
        heuristic_policy = CentralityHeuristicPolicy(policy, weights).to(device=device)
        
        # Custom rollout for heuristic that uses compute_actions
        x0 = sample_initial(case, mc_paths, weights.shape[0], state_dim, device)
        node = highest_degree_node(weights)
        x0 = apply_initial_shock(x0, [node], abm_config.default_threshold - 0.2)
        initial_defaulted = (x0 < abm_config.default_threshold).any(dim=-1)
        
        batch_size, agents, _ = x0.shape
        batched_weights = expand_graph_weights(weights, batch_size)
        noise = torch.randn(
            abm_config.steps,
            batch_size,
            agents,
            state_dim,
            device=x0.device,
            dtype=x0.dtype,
        )
        states = [x0]
        actions_history = []
        current = x0
        defaulted = default_mask(current, abm_config.default_threshold)
        new_defaults = []
        for step in range(abm_config.steps):
            time = step * abm_config.dt
            with torch.inference_mode():
                actions = heuristic_policy.compute_actions(time, current, batched_weights)
            next_states = euler_step(current, abm_config, batched_weights, actions=actions, noise=noise[step])
            states.append(next_states)
            actions_history.append(actions)
            new_def = default_mask(next_states, abm_config.default_threshold) & (~defaulted)
            new_defaults.append(new_def)
            defaulted = defaulted | new_def
            current = next_states
        
        states_tensor = torch.stack(states, dim=0)
        actions_tensor = torch.stack(actions_history, dim=0)
        new_defaults_tensor = torch.stack(new_defaults, dim=0)
        from mfnn_control import ABMSimulationResult
        heuristic_sim = ABMSimulationResult(
            states=states_tensor,
            actions=actions_tensor,
            defaulted=defaulted,
            new_defaults_per_step=new_defaults_tensor,
        )
        heuristic_result = cascade_stats(heuristic_sim, abm_config.default_threshold, initial_defaulted)
        add_cascade_ci(heuristic_result)
        heuristic = {"hub_node": node, "stats": heuristic_result}
        
        u_rate = float(uncontrolled["stats"]["cascade_rate"])
        c_rate = float(controlled["stats"]["cascade_rate"])
        h_rate = float(heuristic["stats"]["cascade_rate"])
        u_size = float(uncontrolled["stats"]["mean_cascade_size"])
        c_size = float(controlled["stats"]["mean_cascade_size"])
        h_size = float(heuristic["stats"]["mean_cascade_size"])
        
        out[name] = {
            "uncontrolled": uncontrolled,
            "controlled": controlled,
            "heuristic": heuristic,
            "cascade_reduction": float(u_rate - c_rate),
            "cascade_size_reduction": float(u_size - c_size),
            "heuristic_cascade_reduction": float(u_rate - h_rate),
            "heuristic_cascade_size_reduction": float(u_size - h_size),
        }
    base_rate = out["homogeneous"]["cascade_reduction"]
    base_size = out["homogeneous"]["cascade_size_reduction"]
    out["mismatch_vs_homogeneous"] = {
        k: {
            "rate": float(base_rate - out[k]["cascade_reduction"]),
            "size": float(base_size - out[k]["cascade_size_reduction"]),
        }
        for k in ("core_periphery", "erdos_renyi")
    }
    heuristic_base_rate = out["homogeneous"]["heuristic_cascade_reduction"]
    heuristic_base_size = out["homogeneous"]["heuristic_cascade_size_reduction"]
    out["heuristic_mismatch_vs_homogeneous"] = {
        k: {
            "rate": float(heuristic_base_rate - out[k]["heuristic_cascade_reduction"]),
            "size": float(heuristic_base_size - out[k]["heuristic_cascade_size_reduction"]),
        }
        for k in ("core_periphery", "erdos_renyi")
    }
    return out


def policy_proxy_cost(sim_result: object, dt: float) -> float:
    actions = sim_result.actions
    control_cost = dt * actions.square().mean(dim=(1, 2, 3)).sum().item()
    terminal_defaults = sim_result.defaulted.float().mean().item()
    return float(control_cost + terminal_defaults)


def experiment_3_limit_breakdown(
    base_config: ABMConfig,
    n_grid: list[int],
    q_grid: list[float],
    case: str,
    mc_paths: int,
    core_hubs: int,
    er_p: float,
    state_dim: int,
    device: str,
    policy: torch.nn.Module,
) -> dict[str, object]:
    by_n = []
    for n in n_grid:
        topologies = build_topologies(n, core_hubs, er_p, device)
        n_item = {"N": n, "topologies": {}}
        for name, weights in topologies.items():
            x0 = sample_initial(case, mc_paths, n, state_dim, device)
            x0 = apply_initial_shock(x0, [highest_degree_node(weights)], base_config.default_threshold - 0.2)
            initial_defaulted = (x0 < base_config.default_threshold).any(dim=-1)
            unc = rollout_abm(x0, base_config, weights, policy=None)
            ctl = rollout_abm(x0, base_config, weights, policy=policy)
            unc_final = unc.defaulted.float().sum(dim=1)
            ctl_final = ctl.defaulted.float().sum(dim=1)
            base_count = initial_defaulted.float().sum(dim=1)
            unc_additional = (unc_final - base_count).clamp_min(0.0)
            ctl_additional = (ctl_final - base_count).clamp_min(0.0)
            unc_dist = [float(v) for v in unc_additional.cpu().tolist()]
            ctl_dist = [float(v) for v in ctl_additional.cpu().tolist()]
            unc_stats: dict = {
                "uncontrolled_cascade_rate": float((unc_additional > 0).float().mean().item()),
                "controlled_cascade_rate": float((ctl_additional > 0).float().mean().item()),
                "uncontrolled_mean_cascade_size": float(unc_additional.mean().item()),
                "controlled_mean_cascade_size": float(ctl_additional.mean().item()),
                "controlled_proxy_cost": policy_proxy_cost(ctl, base_config.dt),
                "uncontrolled_cascade_size_distribution": unc_dist,
                "controlled_cascade_size_distribution": ctl_dist,
            }
            # bootstrap CIs
            u_lo, u_hi = bootstrap_ci(unc_dist, stat="mean")
            c_lo, c_hi = bootstrap_ci(ctl_dist, stat="mean")
            ur_lo, ur_hi = bootstrap_ci(unc_dist, stat="rate")
            cr_lo, cr_hi = bootstrap_ci(ctl_dist, stat="rate")
            unc_stats["uncontrolled_mean_cascade_size_ci"] = [u_lo, u_hi]
            unc_stats["controlled_mean_cascade_size_ci"] = [c_lo, c_hi]
            unc_stats["uncontrolled_cascade_rate_ci"] = [ur_lo, ur_hi]
            unc_stats["controlled_cascade_rate_ci"] = [cr_lo, cr_hi]
            n_item["topologies"][name] = unc_stats
        by_n.append(n_item)

    by_q = []
    for q in q_grid:
        config_q = ABMConfig(
            horizon=base_config.horizon,
            steps=base_config.steps,
            sigma=base_config.sigma,
            interaction_q=q,
            state_dim=base_config.state_dim,
            default_threshold=base_config.default_threshold,
        )
        topologies = build_topologies(int(n_grid[-1]), core_hubs, er_p, device)
        q_item = {"q": float(q), "topologies": {}}
        for name, weights in topologies.items():
            x0 = sample_initial(case, mc_paths, weights.shape[0], state_dim, device)
            x0 = apply_initial_shock(x0, [highest_degree_node(weights)], config_q.default_threshold - 0.2)
            initial_defaulted = (x0 < config_q.default_threshold).any(dim=-1)
            unc = rollout_abm(x0, config_q, weights, policy=None)
            ctl = rollout_abm(x0, config_q, weights, policy=policy)
            unc_final = unc.defaulted.float().sum(dim=1)
            ctl_final = ctl.defaulted.float().sum(dim=1)
            base_count = initial_defaulted.float().sum(dim=1)
            unc_additional = (unc_final - base_count).clamp_min(0.0)
            ctl_additional = (ctl_final - base_count).clamp_min(0.0)
            unc_dist_q = [float(v) for v in unc_additional.cpu().tolist()]
            ctl_dist_q = [float(v) for v in ctl_additional.cpu().tolist()]
            q_row: dict = {
                "uncontrolled_cascade_rate": float((unc_additional > 0).float().mean().item()),
                "controlled_cascade_rate": float((ctl_additional > 0).float().mean().item()),
                "uncontrolled_mean_cascade_size": float(unc_additional.mean().item()),
                "controlled_mean_cascade_size": float(ctl_additional.mean().item()),
            }
            qu_lo, qu_hi = bootstrap_ci(unc_dist_q, stat="mean")
            qc_lo, qc_hi = bootstrap_ci(ctl_dist_q, stat="mean")
            q_row["uncontrolled_mean_cascade_size_ci"] = [qu_lo, qu_hi]
            q_row["controlled_mean_cascade_size_ci"] = [qc_lo, qc_hi]
            q_item["topologies"][name] = q_row
        by_q.append(q_item)
    return {"vs_N": by_n, "vs_q": by_q}


def experiment_4_phase_transition(
    base_config: ABMConfig,
    agents: int,
    q_sweep: list[float],
    case: str,
    mc_paths: int,
    state_dim: int,
    device: str,
    policy: torch.nn.Module,
    topologies: dict[str, torch.Tensor],
) -> dict[str, object]:
    results_by_topology: dict[str, object] = {}
    for topo_name, weights in topologies.items():
        uncontrolled = []
        controlled = []
        hub = highest_degree_node(weights)
        for q in q_sweep:
            cfg = ABMConfig(
                horizon=base_config.horizon,
                steps=base_config.steps,
                sigma=base_config.sigma,
                interaction_q=q,
                state_dim=state_dim,
                default_threshold=base_config.default_threshold,
            )
            x0 = sample_initial(case, mc_paths, agents, state_dim, device)
            x0 = apply_initial_shock(x0, [hub], cfg.default_threshold - 0.2)
            initial_defaulted = (x0 < cfg.default_threshold).any(dim=-1)
            sim_unc = rollout_abm(x0, cfg, weights, policy=None)
            sim_ctl = rollout_abm(x0, cfg, weights, policy=policy)
            unc_add = (sim_unc.defaulted.float().sum(dim=1) - initial_defaulted.float().sum(dim=1)).clamp_min(0.0)
            ctl_add = (sim_ctl.defaulted.float().sum(dim=1) - initial_defaulted.float().sum(dim=1)).clamp_min(0.0)
            unc_dist = [float(v) for v in unc_add.cpu().tolist()]
            ctl_dist = [float(v) for v in ctl_add.cpu().tolist()]
            uncontrolled.append({
                "q": float(q),
                "cascade_rate": float((unc_add > 0).float().mean().item()),
                "mean_cascade_size": float(unc_add.mean().item()),
                "cascade_size_distribution": unc_dist,
                "tail_gof": fit_cascade_tail_gof(unc_dist),
            })
            controlled.append({
                "q": float(q),
                "cascade_rate": float((ctl_add > 0).float().mean().item()),
                "mean_cascade_size": float(ctl_add.mean().item()),
                "cascade_size_distribution": ctl_dist,
                "tail_gof": fit_cascade_tail_gof(ctl_dist),
            })
        unc_rates = [row["cascade_rate"] for row in uncontrolled]
        ctl_rates = [row["cascade_rate"] for row in controlled]
        half = 0.5
        critical_q_unc = next((q_sweep[i] for i, r in enumerate(unc_rates) if r >= half), None)
        critical_q_ctl = next((q_sweep[i] for i, r in enumerate(ctl_rates) if r >= half), None)
        results_by_topology[topo_name] = {
            "uncontrolled": uncontrolled,
            "controlled": controlled,
            "critical_q_uncontrolled": critical_q_unc,
            "critical_q_controlled": critical_q_ctl,
        }
    return results_by_topology


def experiment_5_subcritical_rate_comparison(
    base_config: ABMConfig,
    sigma_values: list[float],
    q_sweep: list[float],
    agents: int,
    case: str,
    mc_paths: int,
    state_dim: int,
    device: str,
    policy: torch.nn.Module,
) -> list[dict[str, object]]:
    """Sweep sigma x q to expose policy rate-reduction in sub/near-critical regime."""
    weights = homogeneous_graph_weights(agents, device=device)
    hub = highest_degree_node(weights)
    results = []
    for sigma in sigma_values:
        cfg_base = ABMConfig(
            horizon=base_config.horizon,
            steps=base_config.steps,
            sigma=sigma,
            interaction_q=q_sweep[0],
            state_dim=state_dim,
            default_threshold=base_config.default_threshold,
        )
        q_rows = []
        for q in q_sweep:
            cfg = ABMConfig(
                horizon=cfg_base.horizon,
                steps=cfg_base.steps,
                sigma=sigma,
                interaction_q=q,
                state_dim=state_dim,
                default_threshold=cfg_base.default_threshold,
            )
            x0 = sample_initial(case, mc_paths, agents, state_dim, device)
            x0 = apply_initial_shock(x0, [hub], cfg.default_threshold - 0.2)
            initial_defaulted = (x0 < cfg.default_threshold).any(dim=-1)
            unc = rollout_abm(x0, cfg, weights, policy=None)
            ctl = rollout_abm(x0, cfg, weights, policy=policy)
            unc_add = (unc.defaulted.float().sum(dim=1) - initial_defaulted.float().sum(dim=1)).clamp_min(0.0)
            ctl_add = (ctl.defaulted.float().sum(dim=1) - initial_defaulted.float().sum(dim=1)).clamp_min(0.0)
            u_rate = float((unc_add > 0).float().mean().item())
            c_rate = float((ctl_add > 0).float().mean().item())
            q_rows.append({
                "q": float(q),
                "uncontrolled_cascade_rate": u_rate,
                "controlled_cascade_rate": c_rate,
                "uncontrolled_mean_cascade_size": float(unc_add.mean().item()),
                "controlled_mean_cascade_size": float(ctl_add.mean().item()),
                "rate_reduction": float(u_rate - c_rate),
            })
        # identify critical q as first q where uncontrolled rate >= 0.5
        cq_unc = next((row["q"] for row in q_rows if row["uncontrolled_cascade_rate"] >= 0.5), None)
        cq_ctl = next((row["q"] for row in q_rows if row["controlled_cascade_rate"] >= 0.5), None)
        results.append({
            "sigma": float(sigma),
            "q_sweep": q_rows,
            "critical_q_uncontrolled": cq_unc,
            "critical_q_controlled": cq_ctl,
        })
    return results


def experiment_wasserstein_convergence(
    base_config: ABMConfig,
    n_grid: list[int],
    case: str,
    mc_paths: int,
    state_dim: int,
    device: str,
    policy: torch.nn.Module,
) -> list[dict]:
    """Compute W1(empirical_N, empirical_N_ref) for each N on the homogeneous topology.

    Uses N_ref = max(n_grid) (i.e. N=1000) as proxy for the MF limit, following
    Fournier & Guillin (2015). The expected rate in d=1 is O(N^{-1/2}).
    """
    try:
        from scipy.stats import wasserstein_distance as w1_distance
    except ImportError:
        return []

    n_ref = n_grid[-1]
    ref_weights = homogeneous_graph_weights(n_ref, device=device)
    torch.manual_seed(42)
    x0_ref = sample_initial(case, mc_paths, n_ref, state_dim, device)
    x0_ref = apply_initial_shock(x0_ref, [highest_degree_node(ref_weights)], base_config.default_threshold - 0.2)

    ref_unc = rollout_abm(x0_ref, base_config, ref_weights, policy=None)
    ref_ctl = rollout_abm(x0_ref, base_config, ref_weights, policy=policy)
    ref_unc_flat = ref_unc.states[-1, :, :, 0].cpu().numpy().flatten()
    ref_ctl_flat = ref_ctl.states[-1, :, :, 0].cpu().numpy().flatten()

    results = []
    for n in n_grid:
        if n == n_ref:
            results.append({"N": n, "w1_uncontrolled": 0.0, "w1_controlled": 0.0})
            continue
        w_n = homogeneous_graph_weights(n, device=device)
        torch.manual_seed(n)
        x0_n = sample_initial(case, mc_paths, n, state_dim, device)
        x0_n = apply_initial_shock(x0_n, [highest_degree_node(w_n)], base_config.default_threshold - 0.2)
        sim_unc = rollout_abm(x0_n, base_config, w_n, policy=None)
        sim_ctl = rollout_abm(x0_n, base_config, w_n, policy=policy)
        unc_flat = sim_unc.states[-1, :, :, 0].cpu().numpy().flatten()
        ctl_flat = sim_ctl.states[-1, :, :, 0].cpu().numpy().flatten()
        results.append({
            "N": n,
            "w1_uncontrolled": float(w1_distance(unc_flat, ref_unc_flat)),
            "w1_controlled": float(w1_distance(ctl_flat, ref_ctl_flat)),
        })
    return results


def run_all_experiments(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    device = args.device
    checkpoint_path = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train a model first by running:\n"
            "  python scripts/train_systemic_risk_baseline.py --algorithm global_bsde --save-dir results/checkpoints"
        )
    policy = load_policy_from_checkpoint(checkpoint_path, device)
    topologies = build_topologies(args.agents, args.core_hubs, args.er_p, device)
    q_values = torch.linspace(args.q_min, args.q_max, args.q_steps).tolist()
    base_config = ABMConfig(
        horizon=args.horizon,
        steps=args.steps,
        sigma=args.sigma,
        interaction_q=args.q_min,
        state_dim=args.state_dim,
        default_threshold=args.default_threshold,
    )
    calibration = calibrate_q(
        base_config,
        topologies["homogeneous"],
        args.case,
        args.mc_paths,
        args.state_dim,
        device,
        q_values,
        args.target_cascade_rate,
    )
    calibrated_config = ABMConfig(
        horizon=base_config.horizon,
        steps=base_config.steps,
        sigma=base_config.sigma,
        interaction_q=float(calibration["selected_q"]),
        state_dim=base_config.state_dim,
        default_threshold=base_config.default_threshold,
    )
    exp1 = experiment_1_uncontrolled(
        calibrated_config,
        topologies,
        args.case,
        args.mc_paths,
        args.state_dim,
        device,
    )
    exp2 = experiment_2_controlled(
        calibrated_config,
        topologies,
        args.case,
        args.mc_paths,
        args.state_dim,
        device,
        policy,
    )
    exp3 = experiment_3_limit_breakdown(
        calibrated_config,
        _to_int_list(args.n_grid),
        _to_float_list(args.q_grid),
        args.case,
        args.mc_paths,
        args.core_hubs,
        args.er_p,
        args.state_dim,
        device,
        policy,
    )
    phase_q_sweep = torch.linspace(args.phase_q_min, args.phase_q_max, args.phase_q_steps).tolist()
    exp4 = experiment_4_phase_transition(
        calibrated_config,
        args.agents,
        phase_q_sweep,
        args.case,
        args.mc_paths,
        args.state_dim,
        device,
        policy,
        topologies,
    )
    subcrit_q_sweep = torch.linspace(args.subcrit_q_min, args.subcrit_q_max, args.subcrit_q_steps).tolist()
    exp5 = experiment_5_subcritical_rate_comparison(
        calibrated_config,
        _to_float_list(args.subcrit_sigmas),
        subcrit_q_sweep,
        args.agents,
        args.case,
        args.mc_paths,
        args.state_dim,
        device,
        policy,
    )
    exp_wasserstein = experiment_wasserstein_convergence(
        calibrated_config,
        _to_int_list(args.n_grid),
        args.case,
        args.mc_paths,
        args.state_dim,
        device,
        policy,
    )
    return {
        "config": {
            "args": vars(args),
            "abm_config": asdict(calibrated_config),
            "checkpoint": str(checkpoint_path),
        },
        "calibration": calibration,
        "experiment_1_uncontrolled": exp1,
        "experiment_2_controlled": exp2,
        "experiment_3_limit_breakdown": exp3,
        "experiment_4_phase_transition": exp4,
        "experiment_5_subcritical_rate_comparison": exp5,
        "experiment_wasserstein_convergence": exp_wasserstein,
    }


def main() -> None:
    args = parse_args()
    output = run_all_experiments(args)
    out_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "abm_experiments.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out_path), "calibrated_q": output["calibration"]["selected_q"]}, indent=2))


if __name__ == "__main__":
    main()