import matplotlib.pyplot as plt
from data import generate_synthetic_svm_data, load_breast_cancer_binary
from admm import admm_svm
from bcd import bcd_svm
from proposed import proposed_svm

# =====================================================================
# PLOTTING HELPER FUNCTIONS
# =====================================================================
def plot_individual_method(method_name, t_syn, obj_syn, acc_syn, t_bc, obj_bc, acc_bc):
    """Generates a 1x3 plot for a single method showing both datasets."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{method_name} Performance Profile', fontsize=16)

    # 1. Objective vs Runtime
    axs[0].plot(t_syn, obj_syn, '-', color='blue', label='Synthetic')
    axs[0].plot(t_bc, obj_bc, '-', color='orange', label='Breast Cancer')
    axs[0].set_title('Training Objective vs. Runtime')
    axs[0].set_xlabel('Runtime (seconds)')
    axs[0].set_ylabel('Objective Value')
    axs[0].set_yscale('log')
    axs[0].grid(True, ls="--")
    axs[0].legend()

    # 2. Objective vs Iteration
    axs[1].plot(range(len(obj_syn)), obj_syn, '-', color='blue', label='Synthetic')
    axs[1].plot(range(len(obj_bc)), obj_bc, '-', color='orange', label='Breast Cancer')
    axs[1].set_title('Training Objective vs. Iteration/Epoch')
    axs[1].set_xlabel('Iteration / Epoch')
    axs[1].set_ylabel('Objective Value')
    axs[1].set_yscale('log')
    axs[1].grid(True, ls="--")
    axs[1].legend()

    # 3. Test Accuracy vs Runtime
    axs[2].plot(t_syn, acc_syn, '-', color='blue', label='Synthetic')
    axs[2].plot(t_bc, acc_bc, '-', color='orange', label='Breast Cancer')
    axs[2].set_title('Test Accuracy vs. Runtime')
    axs[2].set_xlabel('Runtime (seconds)')
    axs[2].set_ylabel('Test Accuracy')
    axs[2].grid(True, ls="--")
    axs[2].legend()

    plt.tight_layout()
    filename = f"{method_name.lower().replace(' ', '_')}_results.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved individual plot: {filename}")

def plot_dataset_comparison(dataset_name,
                            t_adm, obj_adm, acc_adm,
                            t_bcd, obj_bcd, acc_bcd,
                            t_prop, obj_prop, acc_prop):
    """Generates the required 1x3 comparison plot for a specific dataset."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Algorithm Comparison: {dataset_name} Dataset', fontsize=16)

    # 1. Objective vs Runtime
    axs[0].plot(t_adm, obj_adm, '-', color='darkblue', label='ADMM')
    axs[0].plot(t_bcd, obj_bcd, '--', color='deepskyblue', label='BCD (Baseline)')
    axs[0].plot(t_prop, obj_prop, '-.', color='magenta', label='Proposed (Smoothed Newton)')
    axs[0].set_title('Training Objective vs. Runtime')
    axs[0].set_xlabel('Runtime (seconds)')
    axs[0].set_ylabel('Objective Value')
    axs[0].set_yscale('log')
    axs[0].grid(True, which="both", ls="--")
    axs[0].legend()

    # 2. Objective vs Iteration
    axs[1].plot(range(len(obj_adm)), obj_adm, '-', color='darkblue', label='ADMM')
    axs[1].plot(range(len(obj_bcd)), obj_bcd, '--', color='deepskyblue', label='BCD (Baseline)')
    axs[1].plot(range(len(obj_prop)), obj_prop, '-.', color='magenta', label='Proposed (Smoothed Newton)')
    axs[1].set_title('Training Objective vs. Iteration/Epoch')
    axs[1].set_xlabel('Iteration / Epoch')
    axs[1].set_ylabel('Objective Value')
    axs[1].set_yscale('log')
    axs[1].grid(True, which="both", ls="--")
    axs[1].legend()

    # 3. Test Accuracy vs Runtime
    axs[2].plot(t_adm, acc_adm, '-', color='darkblue', label='ADMM')
    axs[2].plot(t_bcd, acc_bcd, '--', color='deepskyblue', label='BCD (Baseline)')
    axs[2].plot(t_prop, acc_prop, '-.', color='magenta', label='Proposed (Smoothed Newton)')
    axs[2].set_title('Test Accuracy vs. Runtime')
    axs[2].set_xlabel('Runtime (seconds)')
    axs[2].set_ylabel('Test Accuracy')
    axs[2].grid(True, ls="--")
    axs[2].legend()

    plt.tight_layout()
    filename = f"comparison_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved comparison plot: {filename}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("Loading datasets...")
    X_tr_syn, y_tr_syn, X_te_syn, y_te_syn, _ = generate_synthetic_svm_data()
    X_tr_bc, y_tr_bc, X_te_bc, y_te_bc = load_breast_cancer_binary()

    lambda_param = 1.0

    # --- 1. RUN ADMM ---
    print("\nRunning ADMM...")
    _, _, obj_adm_syn, acc_adm_syn, t_adm_syn, iter_adm_syn = admm_svm(X_tr_syn, y_tr_syn, X_te_syn, y_te_syn, lambda_=lambda_param)
    _, _, obj_adm_bc, acc_adm_bc, t_adm_bc, iter_adm_bc = admm_svm(X_tr_bc, y_tr_bc, X_te_bc, y_te_bc, lambda_=lambda_param)

    # --- 2. RUN BCD (BASELINE) ---
    print("Running BCD (Baseline)...")
    _, _, obj_bcd_syn, acc_bcd_syn, t_bcd_syn, iter_bcd_syn = bcd_svm(X_tr_syn, y_tr_syn, X_te_syn, y_te_syn, lambda_=lambda_param)
    _, _, obj_bcd_bc, acc_bcd_bc, t_bcd_bc, iter_bcd_bc = bcd_svm(X_tr_bc, y_tr_bc, X_te_bc, y_te_bc, lambda_=lambda_param)

    # --- 3. RUN PROPOSED ALGORITHM ---
    print("Running Proposed Method...")
    _, _, obj_prop_syn, acc_prop_syn, t_prop_syn, iter_prop_syn = proposed_svm(X_tr_syn, y_tr_syn, X_te_syn, y_te_syn, lambda_=lambda_param)
    _, _, obj_prop_bc, acc_prop_bc, t_prop_bc, iter_prop_bc = proposed_svm(X_tr_bc, y_tr_bc, X_te_bc, y_te_bc, lambda_=lambda_param)

    # --- 4. GENERATE PLOTS ---
    print("\nGenerating visual reports...")

    # Individual Profiles
    plot_individual_method("ADMM", t_adm_syn, obj_adm_syn, acc_adm_syn, t_adm_bc, obj_adm_bc, acc_adm_bc)
    plot_individual_method("BCD", t_bcd_syn, obj_bcd_syn, acc_bcd_syn, t_bcd_bc, obj_bcd_bc, acc_bcd_bc)
    plot_individual_method("Proposed", t_prop_syn, obj_prop_syn, acc_prop_syn, t_prop_bc, obj_prop_bc, acc_prop_bc)

    # Required Dataset Comparisons
    plot_dataset_comparison("Synthetic",
                            t_adm_syn, obj_adm_syn, acc_adm_syn,
                            t_bcd_syn, obj_bcd_syn, acc_bcd_syn,
                            t_prop_syn, obj_prop_syn, acc_prop_syn)

    plot_dataset_comparison("Breast Cancer",
                            t_adm_bc, obj_adm_bc, acc_adm_bc,
                            t_bcd_bc, obj_bcd_bc, acc_bcd_bc,
                            t_prop_bc, obj_prop_bc, acc_prop_bc)

    # --- 5. GENERATE REQUIRED TABLES ---
    print("\nGenerating final summary tables...")

    def print_summary_table(dataset_name, results):
        print(f"\nTable: Summary for {dataset_name} Dataset")
        print("-" * 80)
        print(f"{'Method':<20} | {'Runtime (s)':<12} | {'Iterations':<10} | {'Final Obj':<12} | {'Test Acc':<10}")
        print("-" * 80)
        for name, t, obj, acc, iters in results:
            print(f"{name:<20} | {t[-1]:<12.4f} | {iters:<10} | {obj[-1]:<12.4f} | {acc[-1]:<10.4f}")
        print("-" * 80)

    syn_results = [
        ("ADMM", t_adm_syn, obj_adm_syn, acc_adm_syn, iter_adm_syn),
        ("BCD", t_bcd_syn, obj_bcd_syn, acc_bcd_syn, iter_bcd_syn),
        ("Proposed method", t_prop_syn, obj_prop_syn, acc_prop_syn, iter_prop_syn)
    ]

    bc_results = [
        ("ADMM", t_adm_bc, obj_adm_bc, acc_adm_bc, iter_adm_bc),
        ("BCD", t_bcd_bc, obj_bcd_bc, acc_bcd_bc, iter_bcd_bc),
        ("Proposed method", t_prop_bc, obj_prop_bc, acc_prop_bc, iter_prop_bc)
    ]

    print_summary_table("Synthetic", syn_results)
    print_summary_table("Breast Cancer", bc_results)

    print("\nAll tasks completed successfully!")
