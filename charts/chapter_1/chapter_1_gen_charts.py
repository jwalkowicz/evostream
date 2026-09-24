import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "font.family": "serif",
    }
)

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.tight_layout(pad=4.0)

t = np.linspace(0, 100, 1000)


def format_ax(ax, title):
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlim(0, 100)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["C1", "C2"])
    ax.set_xticks([])
    ax.set_xlabel("Czas", loc="right")
    ax.set_ylabel("Klasa")
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# sudden drift
y_sudden = np.where(t < 50, 0, 1)
axs[0, 0].plot(t, y_sudden, color="black", linewidth=1.5)
format_ax(axs[0, 0], "(a) Dryf nagły")

# incremental drift
y_incremental = np.clip(t / 100, 0, 1)
axs[0, 1].plot(t, y_incremental, color="black", linewidth=1.5)
format_ax(axs[0, 1], "(b) Dryf inkrementalny")

# gradual drift
y_gradual = np.zeros_like(t)
y_gradual[(t > 30) & (t < 40)] = 1
y_gradual[(t > 50) & (t < 65)] = 1
y_gradual[t >= 75] = 1
axs[0, 2].plot(t, y_gradual, color="black", linewidth=1.5)
format_ax(axs[0, 2], "(c) Dryf stopniowy")

# recurring drift
y_recurring = np.where((t > 30) & (t < 70), 1, 0)
axs[1, 0].plot(t, y_recurring, color="black", linewidth=1.5)
format_ax(axs[1, 0], "(d) Koncepcje powracające")

# blip (outlier)
y_blip = np.zeros_like(t)
y_blip[(t > 45) & (t < 55)] = np.piecewise(
    t[(t > 45) & (t < 55)],
    [t[(t > 45) & (t < 55)] < 50, t[(t > 45) & (t < 55)] >= 50],
    [lambda x: (x - 45) / 5, lambda x: 1 - (x - 50) / 5],
)
axs[1, 1].plot(t, y_blip, color="black", linewidth=1.5)
format_ax(axs[1, 1], "(e) Zakłócenie")

# noise
np.random.seed(42)
y_noise = 0.5 + 0.05 * np.random.randn(len(t)) + 0.05 * np.sin(t / 2)
axs[1, 2].plot(t, y_noise, color="black", linewidth=1)
format_ax(axs[1, 2], "(f) Szum")

plt.savefig("charts/chapter_1/concept_drift_types.pdf", bbox_inches="tight", format="pdf")
plt.show()
