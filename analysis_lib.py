import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from numba import njit, prange
from sim_lib import simulate_games_detailed
import pyvista as pv

def run_LRB_grid_search(Lmin, Lmax, Rmin, Rmax, Bmin, Bmax,
                        N, t_land, t_ramp, t_bomb, N_sim,
                        gameplan, extra_mulligan=True):
    results = []
    best_rate = -1.0
    best_config = None
    best_fail_summary = None

    total_combos = (Lmax - Lmin + 1) * (Rmax - Rmin + 1) * (Bmax - Bmin + 1)
    combo_counter = 0
    start_time = time.time()

    print(f"\n🚀 Starting optimization over {total_combos:,} combinations...\n")

    for L in range(Lmin, Lmax + 1):
        for R in range(Rmin, Rmax + 1):
            for B in range(Bmin, Bmax + 1):

                if L + R + B > N:
                    continue

                combo_counter += 1

                # Build deck
                base_deck = np.concatenate([
                    np.ones(L, dtype=np.uint8),
                    np.full(R, 2, dtype=np.uint8),
                    np.full(B, 3, dtype=np.uint8),
                    np.zeros(N - L - R - B, dtype=np.uint8)
                ])

                # Run simulation
                result, fail_summary, total_failures = simulate_games_detailed(
                    N_sim, base_deck, t_land, t_ramp, t_bomb, gameplan, extra_mulligan
                )

                results.append((L, R, B, result))

                # Track best result
                if result > best_rate:
                    best_rate = result
                    best_config = (L, R, B)
                    best_fail_summary = fail_summary

                if combo_counter % 100 == 0 or combo_counter == total_combos:
                    elapsed = time.time() - start_time
                    print(f"Checked {combo_counter}/{total_combos} configs "
                          f"(Best: L={best_config[0]}, R={best_config[1]}, B={best_config[2]} → {best_rate:.2f}%) "
                          f"Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n🏁 Optimization complete in {elapsed:.1f}s.")
    print(f"🔹 Best configuration: Lands={best_config[0]}, Ramp={best_config[1]}, Bombs={best_config[2]}")
    print(f"🔹 Success rate: {best_rate:.2f}%")

    data = np.array(results) #structure (total_combos, 4); each column is L, R, B, success_rate
    return data, best_config, best_rate, best_fail_summary

# ============================================================
# 🔹 Helper: Sensitivity Analysis
# ============================================================
def analyze_sensitivity(data):
    L_vals, R_vals, B_vals, rates = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    L_unique, L_avg = marginal_effect(L_vals, rates)
    R_unique, R_avg = marginal_effect(R_vals, rates)
    B_unique, B_avg = marginal_effect(B_vals, rates)

    L_slope = slope(L_unique, L_avg)
    R_slope = slope(R_unique, R_avg)
    B_slope = slope(B_unique, B_avg)

    slopes = {"Lands": L_slope, "Ramp": R_slope, "Bombs": B_slope}
    dominant = max(slopes, key=lambda k: abs(slopes[k]))

    print("\n🔍 Sensitivity Summary (ΔSuccessRate per +1 card):")
    print(f"  Lands (L): {L_slope:+.3f}% per +1")
    print(f"  Ramp  (R): {R_slope:+.3f}% per +1")
    print(f"  Bombs (B): {B_slope:+.3f}% per +1")
    print(f"🏆 Strongest influence: {dominant} ({slopes[dominant]:+.3f}% per card)\n")

    return {
        "L_trend": (L_unique, L_avg),
        "R_trend": (R_unique, R_avg),
        "B_trend": (B_unique, B_avg),
        "slopes": slopes,
        "dominant": dominant
    }

def marginal_effect(var_vals, rates):
    unique_vals = np.unique(var_vals)
    avg_rates = np.zeros_like(unique_vals, dtype=np.float64)
    for i, val in enumerate(unique_vals):
        mask = var_vals == val
        avg_rates[i] = np.mean(rates[mask])
    return unique_vals, avg_rates

def slope(vals, avg):
        return (avg[-1] - avg[0]) / (vals[-1] - vals[0]) if len(vals) > 1 else 0.0

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