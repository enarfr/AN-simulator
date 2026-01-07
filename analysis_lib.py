import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from numba import njit, prange
from sim_lib import simulate_games_detailed
# import pyvista as pv

def run_LRBD_grid_search(Lmin, Lmax, Rmin, Rmax, Bmin, Bmax, Dmin, Dmax,
                         N, t_land, t_ramp, t_bomb, t_draw, N_sim,
                         gameplan, extra_mulligan=True):
    """
    Extended grid search including Draw cards (D).
    Loops over all combinations of L, R, B, D and tracks success rates.
    """
    results = []
    best_rate = -1.0
    best_config = None
    best_fail_summary = None

    total_combos = (Lmax - Lmin + 1) * (Rmax - Rmin + 1) * (Bmax - Bmin + 1) * (Dmax - Dmin + 1)
    combo_counter = 0
    start_time = time.time()

    print(f"\n🚀 Starting optimization over {total_combos:,} combinations...\n")

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
                        np.zeros(N - L - R - B - D, dtype=np.uint8)
                    ])

                    # Run simulation
                    result, fail_summary = simulate_games_detailed(
                        N_sim, base_deck, t_land, t_ramp, t_bomb, t_draw, gameplan, extra_mulligan
                    )

                    results.append((L, R, B, D, result))

                    # Track best result
                    if result > best_rate:
                        best_rate = result
                        best_config = (L, R, B, D)
                        best_fail_summary = fail_summary

                    if combo_counter % 100 == 0 or combo_counter == total_combos:
                        elapsed = time.time() - start_time
                        print(f"Checked {combo_counter}/{total_combos} configs "
                              f"(Best: L={best_config[0]}, R={best_config[1]}, B={best_config[2]}, D={best_config[3]} → {best_rate:.2f}%) "
                              f"Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n🏁 Optimization complete in {elapsed:.1f}s.")
    print(f"🔹 Best configuration: Lands={best_config[0]}, Ramp={best_config[1]}, Bombs={best_config[2]}, Draw={best_config[3]}")
    print(f"🔹 Success rate: {best_rate:.2f}%")

    data = np.array(results)  # structure (total_combos, 5); each column = L, R, B, D, success_rate
    return data, best_config, best_rate, best_fail_summary


# ============================================================
# 🔹 Helper: Sensitivity Analysis
# ============================================================
def analyze_sensitivity(data):
    """
    Analyze sensitivity of success rate to each card type (L, R, B, D).
    
    Parameters
    ----------
    data : np.ndarray
        Array with columns [L, R, B, D, success_rate]
    
    Returns
    -------
    dict
        Trends, slopes, and dominant card type.
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
    print(f"  Lands (L): {L_slope:+.3f}% per +1")
    print(f"  Ramp  (R): {R_slope:+.3f}% per +1")
    print(f"  Bombs (B): {B_slope:+.3f}% per +1")
    print(f"  Draws (D): {D_slope:+.3f}% per +1")
    print(f"🏆 Strongest influence: {dominant} ({slopes[dominant]:+.3f}% per card)\n")

    return {
        "L_trend": (L_unique, L_avg),
        "R_trend": (R_unique, R_avg),
        "B_trend": (B_unique, B_avg),
        "D_trend": (D_unique, D_avg),
        "slopes": slopes,
        "dominant": dominant
    }

# The helper functions remain the same:
def marginal_effect(var_vals, rates):
    unique_vals = np.unique(var_vals)
    avg_rates = np.zeros_like(unique_vals, dtype=np.float64)
    for i, val in enumerate(unique_vals):
        mask = var_vals == val
        avg_rates[i] = np.mean(rates[mask])
    return unique_vals, avg_rates

def slope(vals, avg):
    return (avg[-1] - avg[0]) / (vals[-1] - vals[0]) if len(vals) > 1 else 0.0



#TODO: Modify these functions
def export_to_paraview(data, filename_base="success_data"):
    """
    Export a NumPy array with structure (L, R, B, success_rate) to ParaView.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N_points, 4) with columns [L, R, B, success_rate]
    filename_base : str
        Base name for output file (without extension)

    Output
    ------
    Saves either a .vts (structured grid) or .vtu (unstructured grid) file
    compatible with ParaView.
    """
    # Extract columns
    L = data[:, 0]
    R = data[:, 1]
    B = data[:, 2]
    success = data[:, 3]

    # Check if it's a regular grid
    L_unique = np.unique(L)
    R_unique = np.unique(R)
    B_unique = np.unique(B)

    if len(L_unique) * len(R_unique) * len(B_unique) == len(data):
        # Regular structured grid
        print("Detected regular grid. Exporting as .vts (StructuredGrid).")
        # Reshape to 3D arrays
        L_grid = L.reshape(len(L_unique), len(R_unique), len(B_unique))
        R_grid = R.reshape(len(L_unique), len(R_unique), len(B_unique))
        B_grid = B.reshape(len(L_unique), len(R_unique), len(B_unique))
        success_grid = success.reshape(len(L_unique), len(R_unique), len(B_unique))

        grid = pv.StructuredGrid(L_grid, R_grid, B_grid)
        grid["success_rate"] = success_grid.ravel(order="F")
        out_file = f"{filename_base}.vts"
        grid.save(out_file)
        print(f"Saved structured grid to {out_file}")

    else:
        # Irregular / scattered points
        print("Detected irregular points. Exporting as .vtu (UnstructuredGrid).")
        points = data[:, :3]
        grid = pv.PolyData(points)
        grid["success_rate"] = success
        out_file = f"{filename_base}.vtu"
        grid.save(out_file)
        print(f"Saved unstructured grid to {out_file}")