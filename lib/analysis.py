import numpy as np
import time
from lib.simulation import simulate_games, CARD_TYPE_MAP
import matplotlib.pyplot as plt
from pathlib import Path

fs = (4, 2.25) #figsize for plots

def print_simulation_report(result, fail_summary, mulligan_stats, N_sim, N_mulligans):
    """
    Print a formatted report of simulation results, including success rate,
    failure breakdown, and mulligan usage statistics.
    
    Parameters
    ----------
    result : float
        Overall success rate (percentage).
    fail_summary : np.ndarray
        2D array of failure counts by turn and operation.
    mulligan_stats : np.ndarray
        Array recording the number of mulligans performed in each game.
    N_sim : int
        Total number of simulated games.
    N_mulligans : int
        Maximum number of mulligans allowed.
    """
    
    success_rate = result
    fail_rate = 100 - success_rate
    total_failures = np.sum(fail_summary)

    print("\n✅ Success rate: {:.2f}%".format(success_rate))
    print("❌ Failure rate: {:.2f}%".format(fail_rate))

    # --- Mulligan usage breakdown ---
    print("\n🎲 Mulligan usage breakdown:")
    for m in range(N_mulligans + 1):
        count = np.sum(mulligan_stats == m)
        pct = count / N_sim * 100
        print(f"{m} mulligan(s): {count} games → {pct:.2f}%")

    # --- Failure breakdown ---
    operation_labels = [["Play Land"], ["Play Land", "Play Ramp"], ["Play Land"], ["Play Land", "Play Bomb"]]

    print("\n🔎 Failure breakdown by operation (as % of ALL games):")
    for turn in range(len(fail_summary)):
        for op in range(len(operation_labels[turn])):
            print(f" Turn {turn+1} - {operation_labels[turn][op]}: {fail_summary[turn, op] / N_sim * 100:6.3f}%")

    print("\n📊 Conditional breakdown (as % of FAILED games):")
    for turn in range(len(fail_summary)):
        for op in range(len(operation_labels[turn])):
            pct = 0.0
            if total_failures > 0:
                pct = fail_summary[turn, op] / total_failures * 100
            print(f" Turn {turn+1} - {operation_labels[turn][op]}: {pct:6.2f}%")

def LRBD_grid_search(Lmin, Lmax, Rmin, Rmax, Bmin, Bmax, Dmin, Dmax,
                     N, t_land, t_ramp, t_bomb, t_draw, N_sim,
                     gameplan, N_mulligans=1, priority_chars=('O','B','R','L','D')):
    """
    Perform a grid search over deck compositions, evaluating the success rate
    of each combination of Lands (L), Ramp (R), Bombs (B), and Draw cards (D).

    For each valid combination (total cards ≤ N), the function builds a deck, 
    runs a Monte Carlo simulation using `simulate_games` from `sim_lib`, and 
    records the resulting success rate. Tracks the best-performing configuration.

    Parameters
    ----------
    Lmin, Lmax : int
        Minimum and maximum number of Land cards to test.
    Rmin, Rmax : int
        Minimum and maximum number of Ramp cards to test.
    Bmin, Bmax : int
        Minimum and maximum number of Bomb cards to test.
    Dmin, Dmax : int
        Minimum and maximum number of Draw cards to test.
    N : int
        Total deck size.
    t_land, t_ramp, t_bomb, t_draw : int
        Minimum number of each card type required in the opening hand.
    N_sim : int
        Number of Monte Carlo simulations to run per deck configuration.
    gameplan : list of np.ndarray
        List of per-turn required card plays (numeric codes).
    N_mulligans : int, optional
        Maximum number of mulligans allowed during simulations.
    priority_chars : tuple of str, optional
        Mulligan bottoming priority as characters ('O','L','R','B','D').

    Returns
    -------
    data : np.ndarray
        Array of shape (N_configs, 5) with columns [L, R, B, D, success_rate].
    best_config : tuple
        Deck configuration (L, R, B, D) with the highest success rate.
    best_rate : float
        Best observed success rate (%) among all configurations.
    best_fail_summary : np.ndarray
        Failure summary array for the best configuration (per-turn, per-operation).
    """
    
    # Convert user-friendly priority to numeric codes
    numeric_priority = tuple(CARD_TYPE_MAP[c] for c in priority_chars)

    results = []
    best_rate = -1.0
    best_config = None
    best_fail_summary = None

    total_combos = (Lmax - Lmin + 1) * (Rmax - Rmin + 1) * (Bmax - Bmin + 1) * (Dmax - Dmin + 1)
    combo_counter = 0
    start_time = time.time()

    print(f"\n🚀 Starting optimization over {total_combos} combinations...\n")

    for L in range(Lmin, Lmax + 1):
        for R in range(Rmin, Rmax + 1):
            for B in range(Bmin, Bmax + 1):
                for D in range(Dmin, Dmax + 1):

                    combo_counter += 1

                    # Build deck
                    base_deck = np.concatenate([
                        np.full(L, 1, dtype=np.uint8),
                        np.full(R, 2, dtype=np.uint8),
                        np.full(B, 3, dtype=np.uint8),
                        np.full(D, 4, dtype=np.uint8),
                        np.full(N - L - R - B - D, 0, dtype=np.uint8)
                    ])

                    # Run simulation
                    result, fail_summary, mulligan_stats = simulate_games(N_sim, base_deck,
                                                                          t_land, t_ramp, t_bomb, t_draw,
                                                                          gameplan, N_mulligans, numeric_priority)

                    results.append((L, R, B, D, result))

                    # Track best result
                    if result > best_rate:
                        best_rate = result
                        best_config = (L, R, B, D)
                        best_fail_summary = fail_summary

                    if combo_counter % 500 == 0 or combo_counter == total_combos: #check best so far every 500 combinations
                        elapsed = time.time() - start_time
                        print(f"Checked {combo_counter}/{total_combos} configs "
                              f"(Best: L={best_config[0]}, R={best_config[1]}, B={best_config[2]}, D={best_config[3]} → {best_rate:.2f}%) "
                              f"Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n✅ Optimization complete in {elapsed:.1f}s.")
    print(f"🔹 Best configuration: Lands={best_config[0]}, Ramp={best_config[1]}, Bombs={best_config[2]}, Draw={best_config[3]}")
    print(f"🔹 Success rate: {best_rate:.2f}%")

    data = np.array(results)
    return data, best_config, best_rate, best_fail_summary


def analyze_sensitivity(data):
    """
    Analyze the sensitivity of the success rate with respect to each card type.

    Computes marginal trends of success rate for each card type (Lands, Ramp, Bombs, Draw)
    and calculates average slopes (ΔSuccessRate per +1 card). Identifies the card type
    with the strongest influence on deck performance.

    Parameters
    ----------
    data : np.ndarray
        Array with columns [L, R, B, D, success_rate].

    Returns
    -------
    dict
        Dictionary containing:
        - 'L_trend', 'R_trend', 'B_trend', 'D_trend': tuple(unique_values, average success rate)
        - 'slopes': dict of average slope per card type
        - 'dominant': card type with strongest influence
    """
    L_vals, R_vals, B_vals, D_vals, rates = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    L_unique, L_avg = marginal_effect(L_vals, rates)
    R_unique, R_avg = marginal_effect(R_vals, rates)
    B_unique, B_avg = marginal_effect(B_vals, rates)
    D_unique, D_avg = marginal_effect(D_vals, rates)

    L_slope = slope(L_unique, L_avg)
    R_slope = slope(R_unique, R_avg)
    B_slope = slope(B_unique, B_avg)
    D_slope = slope(D_unique, D_avg)

    slopes = {"Lands": L_slope, "Ramp": R_slope, "Bombs": B_slope, "Draw": D_slope}
    dominant = max(slopes, key=lambda k: abs(slopes[k]))

    print("\n🔍 Sensitivity Summary (ΔSuccessRate per +1 card):")
    print(f"  Lands (L): {L_slope:+.3f}% per +1")
    print(f"  Ramp  (R): {R_slope:+.3f}% per +1")
    print(f"  Bombs (B): {B_slope:+.3f}% per +1")
    print(f"  Draws (D): {D_slope:+.3f}% per +1")
    print(f"🏆 Strongest influence: {dominant} ({slopes[dominant]:+.3f}% per card)\n")

    return {"L_trend": (L_unique, L_avg),
            "R_trend": (R_unique, R_avg),
            "B_trend": (B_unique, B_avg),
            "D_trend": (D_unique, D_avg),
            "slopes": slopes,
            "dominant": dominant}


def marginal_effect(var_vals, rates):
    """
    Compute the marginal effect of a single card type on success rate.

    For each unique card count, computes the average success rate across all
    configurations sharing that count.

    Parameters
    ----------
    var_vals : np.ndarray
        Array of card counts for a single type (L, R, B, or D).
    rates : np.ndarray
        Corresponding success rates.

    Returns
    -------
    unique_vals : np.ndarray
        Sorted unique values of the card type.
    avg_rates : np.ndarray
        Average success rate associated with each unique value.
    """
    
    unique_vals = np.unique(var_vals)
    avg_rates = np.zeros_like(unique_vals, dtype=np.float64)
    for i, val in enumerate(unique_vals):
        mask = var_vals == val
        avg_rates[i] = np.mean(rates[mask])
    return unique_vals, avg_rates

def slope(vals, avg):
    """
    Compute the average slope (ΔSuccessRate per +1 card) from discrete points.

    Parameters
    ----------
    vals : np.ndarray
        Independent variable values (card counts).
    avg : np.ndarray
        Dependent variable values (average success rates).

    Returns
    -------
    float
        Estimated slope. Returns 0.0 if only one unique value exists.
    """
    
    if len(vals) > 1:
        dy = avg[-1] - avg[0]
        dx = vals[-1] - vals[0]
        s = dy/dx
    else:
        s = 0
        
    return s

def analyze_single_variable(var_name, var_min, var_max,
                            fixed_config, N,
                            t_land, t_ramp, t_bomb, t_draw,
                            N_sim, gameplan, N_mulligans, priority_chars,
                            show_plot=True, save_plot=False):
    """
    Analyze the effect of varying a single card type on success rate while
    keeping all other card counts fixed. Also computes the numerical derivative
    of the winrate with respect to the variable.

    The resulting plot shows:
    - Winrate (%) vs variable (left y-axis)
    - Marginal gain (ΔWinrate per +1 card) (right y-axis)

    Parameters
    ----------
    var_name : str
        Variable to vary: one of {'L', 'R', 'B', 'D'}.
    var_min, var_max : int
        Range of values (inclusive) for the variable.
    fixed_config : dict
        Fixed values for the other variables.
    N : int
        Total deck size.
    t_land, t_ramp, t_bomb, t_draw : int
        Opening hand requirements.
    N_sim : int
        Number of simulations per configuration.
    gameplan : list of np.ndarray
        Per-turn required plays.
    N_mulligans : int, optional
        Maximum number of mulligans allowed.
    priority_chars : tuple of str, optional
        Mulligan priority using character encoding.
    show_plot : bool, optional
        Whether to display a matplotlib plot.

    Returns
    -------
    values : np.ndarray
        Tested values of the variable.
    winrates : np.ndarray
        Corresponding success rates (%).
    derivatives : np.ndarray
        Numerical derivative (Δwinrate per +1 unit).
    """

    numeric_priority = tuple(CARD_TYPE_MAP[c] for c in priority_chars)

    values = []
    winrates = []

    # --- Run simulations ---
    for val in range(var_min, var_max + 1):

        cfg = fixed_config.copy()
        cfg[var_name] = val

        L, R, B, D = cfg['L'], cfg['R'], cfg['B'], cfg['D']

        if L + R + B + D > N:
            continue

        base_deck = np.concatenate([
            np.ones(L, dtype=np.uint8),
            np.full(R, 2, dtype=np.uint8),
            np.full(B, 3, dtype=np.uint8),
            np.full(D, 4, dtype=np.uint8),
            np.zeros(N - L - R - B - D, dtype=np.uint8)
        ])

        result, _, _ = simulate_games(
            N_sim, base_deck,
            t_land, t_ramp, t_bomb, t_draw,
            gameplan,
            N_mulligans,
            numeric_priority
        )

        values.append(val)
        winrates.append(result)

    values = np.array(values)
    winrates = np.array(winrates)

    # --- Compute derivative of winrate with respect to value --- 
    derivatives = np.array([(winrates[i+1] - winrates[i-1]) / 2.0 for i in range(1, len(winrates) - 1)]) #central differences
    interior_values = values[1:-1] #axis for derivatives

    # --- Plot ---
    if show_plot:
        fig, ax1 = plt.subplots(figsize=(4, 2.25))
    
        # --- Left axis (winrate) ---
        color1 = "tab:blue"
        ax1.set_xlabel(f"{var_name} count", fontsize=6)
        ax1.set_ylabel("Winrate (%)", color=color1, fontsize=6)
        ax1.plot(values, winrates, marker='o', color=color1, markersize=3, linewidth=1)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=5)
        ax1.tick_params(axis='x', labelsize=5)
        ax1.spines['left'].set_color(color1)
        ax1.grid(True, linewidth=0.5)
    
        # --- Right axis (derivative) ---
        color2 = "tab:red"
        ax2 = ax1.twinx()
        ax2.set_ylabel("Δ Winrate per +1 card (%)", color=color2, fontsize=6)
        ax2.plot(interior_values, derivatives, linestyle='--', marker='x', color=color2, markersize=3, linewidth=1)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=5)
        ax2.spines['right'].set_color(color2)
    
        # Title and layout
        plt.title(f"Winrate and Marginal Gain vs {var_name}", fontsize=7)
        #plt.subplots_adjust(left=0.15, right=0.85, top=0.88, bottom=0.18)
        plt.tight_layout(pad=0.3)
        if save_plot:
            fig_name = var_name + "_gain.png"
            save_dir = Path.cwd() /"data"
            plt.savefig(save_dir/fig_name, dpi=300)
            plt.close()
        else:
            plt.show()

    return values, winrates, derivatives