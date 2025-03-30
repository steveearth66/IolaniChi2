import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2
import tkinter as tk

# --- Setup ---
root = tk.Tk()
root.withdraw()  # Hide the root window

categories = ["low", "medlow", "medhigh", "high"]
k = len(categories)
expected = [0.25, 0.25, 0.25, 0.25]
if len(expected) != k or not np.isclose(sum(expected), 1.0):
    raise ValueError("Expected probabilities must sum to 1.")

num_trials = 2000
sample_size = 100
chi_stats = []
trial_counter = 0
run_continuous = False
last_observed = []

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(top=0.85)
hist_bins = np.linspace(0, 20, 50)

def update_plot():
    ax.clear()
    ax.hist(chi_stats, bins=hist_bins, alpha=0.7, label='Simulated Chi-Square Stats')

    if trial_counter >= num_trials:
        x = np.linspace(0, 20, 500)
        df = k - 1
        y = chi2.pdf(x, df) * len(chi_stats) * (hist_bins[1] - hist_bins[0])
        ax.plot(x, y, 'r-', lw=2, label=f'Chi-Square PDF (df={df})')

    ax.set_xlim(hist_bins[0], hist_bins[-1])
    ax.set_ylim(0, max(10, max(np.histogram(chi_stats, bins=hist_bins)[0]) + 1))
    ax.set_xlabel("Chi-Square Statistic")
    ax.set_ylabel("Frequency")
    ax.set_title("Chi-Square Simulation")

    if trial_counter > 0:
        observed_counts = last_observed
        expected_counts = np.array(expected) * sample_size
        terms = [(o, e, (o - e) ** 2 / e) for o, e in zip(observed_counts, expected_counts)]
        chi_statistic = sum(term[2] for term in terms)
        chi_calc_display = " + ".join([f"(({o}-{e:.1f})²/{e:.1f})" for o, e, _ in terms])
        full_display = "\n".join([
            f"Trial {trial_counter}/{num_trials}",
            *(f"{cat}: {count}" for cat, count in zip(categories, observed_counts)),
            f"Chi² = {chi_statistic:.3f}",
            f"= {chi_calc_display}"
        ])
        ax.text(0.02, 0.95, full_display,
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.legend()
    fig.canvas.draw_idle()

# --- Trial logic ---
def run_one_trial():
    global trial_counter, last_observed
    if trial_counter < num_trials:
        observed_counts = np.random.multinomial(sample_size, expected)
        stat, _ = chisquare(observed_counts, f_exp=np.array(expected) * sample_size)
        chi_stats.append(stat)
        trial_counter += 1
        last_observed = observed_counts
        update_plot()

def run_trials_with_after():
    global run_continuous
    if run_continuous and trial_counter < num_trials:
        run_one_trial()
        root.after(50, run_trials_with_after)

# --- Keyboard control ---
def on_key(event):
    global run_continuous
    if event.key == ' ':
        run_one_trial()
    elif event.key == 'x':
        if trial_counter < num_trials:
            run_continuous = True
            root.after(50, run_trials_with_after)

fig.canvas.mpl_connect('key_press_event', on_key)

# --- Start ---
update_plot()
plt.show()