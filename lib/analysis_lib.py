import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from numba import njit, prange
from lib.sim_lib import simulate_games

def LRBD_grid_search(Lmin, Lmax, Rmin, Rmax, Bmin, Bmax, Dmin, Dmax,
                         N, t_land, t_ramp, t_bomb, t_draw, N_sim,
                         gameplan, extra_mulligan=True):
    """
    Perform a grid search over deck compositions including Lands (L), Ramp (R),
    Bombs (B), and Draw cards (D), evaluating the success rate of each configuration.
    
    For each combination of L, R, B, and D, the function builds a deck, runs a Monte 
    Carlo simulation, and records the resulting success rate. The function also tracks and reports the
    best-performing configuration encountered during the search.
    
    Parameters
    ----------
    Lmin, Lmax : int
        Minimum and maximum number of Lands to test.
    Rmin, Rmax : int
        Minimum and maximum number of Ramp cards to test.
    Bmin, Bmax : int
        Minimum and maximum number of Bomb cards to test.
    Dmin, Dmax : int
        Minimum and maximum number of Draw cards to test.
    N : int
        Total deck size.
    t_land : int
        Minimum Lands required in the opening hand.
    t_ramp : int
        Minimum Ramp cards required in the opening hand.
    t_bomb : int
        Minimum Bomb cards required in the opening hand.
    t_draw : int
        Minimum Draw cards required in the opening hand.
    N_sim : int
        Number of simulations per configuration.
    gameplan : list of np.ndarray
        Per-turn list of required card types to be played.
    extra_mulligan : bool, optional
        Whether to enable the priority-based extra mulligan.
    
    Returns
    -------
    data : np.ndarray
        Array of shape (N_configs, 5) with columns [L, R, B, D, success_rate].
    best_config : tuple
        Deck configuration (L, R, B, D) with the highest success rate.
    best_rate : float
        Best observed success rate.
    best_fail_summary : np.ndarray
        Failure summary corresponding to the best configuration.
    """

    results = []
    best_rate = -1.0
    best_config = None
    best_fail_summary = None

    total_combos = (Lmax - Lmin + 1) * (Rmax - Rmin + 1) * (Bmax - Bmin + 1) * (Dmax - Dmin + 1)
    combo_counter = 0
    start_time = time.time()

    print("\n🚀 Starting optimization over {} combinations...\n".format(total_combos))

    for L in range(Lmin, Lmax + 1):
        for R in range(Rmin, Rmax + 1):
            for B in range(Bmin, Bmax + 1):
                for D in range(Dmin, Dmax + 1):

                    if L + R + B + D > N:
                        continue

                    combo_counter += 1

                    # Build deck
                    base_deck = np.concatenate([
                        np.ones(L, dtype=np.uint8),
                        np.full(R, 2, dtype=np.uint8),
                        np.full(B, 3, dtype=np.uint8),
                        np.full(D, 4, dtype=np.uint8),
                        np.zeros(N - L - R - B - D, dtype=np.uint8) ])

                    # Run simulation
                    result, fail_summary, mulligan_stats= simulate_games_detailed(N_sim, base_deck, t_land, t_ramp, t_bomb, t_draw, gameplan, extra_mulligan)

                    results.append((L, R, B, D, result))

                    # Track best result
                    if result > best_rate:
                        best_rate = result
                        best_config = (L, R, B, D)
                        best_fail_summary = fail_summary

                    if combo_counter % 100 == 0 or combo_counter == total_combos:
                        elapsed = time.time() - start_time
                        print("Checked {}/{} configs "
                              "(Best: L={}, R={}, B={}, D={} → {:.2f}%) "
                              "Elapsed: {:.1f}s".format(combo_counter, total_combos, best_config[0], best_config[1], best_config[2], best_config[3], best_rate, elapsed))


    elapsed = time.time() - start_time
    print("\n Optimization complete in {:.1f}s.".format(elapsed))
    print("🔹 Best configuration: Lands={}, Ramp={}, Bombs={}, Draw={}".format(best_config[0], best_config[1], best_config[2], best_config[3]))
    print(f"🔹 Success rate: {best_rate:.2f}%")

    data = np.array(results)  # structure (total_combos, 5); each column = L, R, B, D, success_rate
    return data, best_config, best_rate, best_fail_summary

def analyze_sensitivity(data):
    """
    Analyze the sensitivity of the success rate with respect to each card type:
    Lands (L), Ramp (R), Bombs (B), and Draw cards (D).
    
    The analysis computes marginal success-rate trends and average slopes
    (Δ success rate per additional card) for each card type, identifying which
    type has the strongest influence on performance.
    
    Parameters
    ----------
    data : np.ndarray
        Array with columns [L, R, B, D, success_rate].
    
    Returns
    -------
    dict
        Dictionary containing:
        - Per-card-type trends (unique values and average success rates),
        - Slopes representing marginal effects,
        - The card type with the strongest overall influence.
    """

    # Extract columns
    L_vals, R_vals, B_vals, D_vals, rates = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    # Compute marginal effects for each card type
    L_unique, L_avg = marginal_effect(L_vals, rates)
    R_unique, R_avg = marginal_effect(R_vals, rates)
    B_unique, B_avg = marginal_effect(B_vals, rates)
    D_unique, D_avg = marginal_effect(D_vals, rates)

    # Compute slopes (ΔSuccessRate per +1 card)
    L_slope = slope(L_unique, L_avg)
    R_slope = slope(R_unique, R_avg)
    B_slope = slope(B_unique, B_avg)
    D_slope = slope(D_unique, D_avg)

    slopes = {"Lands": L_slope, "Ramp": R_slope, "Bombs": B_slope, "Draw": D_slope}
    dominant = max(slopes, key=lambda k: abs(slopes[k]))

    print("\n🔍 Sensitivity Summary (ΔSuccessRate per +1 card):")
    print("  Lands (L): {:+.3f}% per +1".format(L_slope))
    print("  Ramp  (R): {:+.3f}% per +1".format(R_slope))
    print("  Bombs (B): {:+.3f}% per +1".format(B_slope))
    print("  Draws (D): {:+.3f}% per +1".format(D_slope))
    print("🏆 Strongest influence: {} ({:+.3f}% per card)\n".format(dominant, slopes[dominant]))

    return {"L_trend": (L_unique, L_avg),
            "R_trend": (R_unique, R_avg),
            "B_trend": (B_unique, B_avg),
            "D_trend": (D_unique, D_avg),
            "slopes": slopes,
            "dominant": dominant}


def marginal_effect(var_vals, rates):
    """
    Compute the marginal effect of a single variable on success rate.
    
    For each unique value of the variable, the function averages the success
    rates of all configurations sharing that value.
    
    Parameters
    ----------
    var_vals : np.ndarray
        Array of values for a single variable (e.g., L, R, B, or D).
    rates : np.ndarray
        Corresponding success rates.
    
    Returns
    -------
    unique_vals : np.ndarray
        Sorted unique values of the variable.
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
    Compute the average slope of a trend line defined by discrete values.
        
    Parameters
    ----------
    vals : np.ndarray
        Independent variable values.
    avg : np.ndarray
        Dependent variable values (e.g., average success rates).
    
    Returns
    -------
    float
        Estimated slope (Δ dependent variable per +1 unit of independent variable).
    """

    return (avg[-1] - avg[0]) / (vals[-1] - vals[0]) if len(vals) > 1 else 0.0