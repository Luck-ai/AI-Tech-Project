import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB


def plot_nb_gaussians(model: GaussianNB,
                      feature_names: list,
                      sample: np.ndarray,
                      class_names: list = None,
                      max_features: int = 20,
                      cols: int = 4,
                      figsize_per_plot=(4, 3)):

    sample = np.asarray(sample).ravel()
    n_features_total = len(feature_names)
    n_features = min(n_features_total, max_features)
    feature_names = feature_names[:n_features]
    sample = sample[:n_features]

    if class_names is None:
        class_names = [str(c) for c in model.classes_]

    n_classes = len(class_names)
    cmap = plt.get_cmap("tab10")
    class_colors = [cmap(i % 10) for i in range(n_classes)]

    rows = int(np.ceil(n_features / cols))
    fig_w = figsize_per_plot[0] * cols
    fig_h = figsize_per_plot[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).flatten()

    for feat_idx in range(n_features):
        ax = axes[feat_idx]
        x_val = sample[feat_idx]

        all_means = model.theta_[:, feat_idx]          
        all_vars  = model.var_[:, feat_idx]           
        all_stds  = np.sqrt(all_vars)

        x_min = min(all_means - 3.5 * all_stds)
        x_max = max(all_means + 3.5 * all_stds)

        x_min = min(x_min, x_val - abs(x_val) * 0.1 - 1e-6)
        x_max = max(x_max, x_val + abs(x_val) * 0.1 + 1e-6)

        xs = np.linspace(x_min, x_max, 400)

        for cls_idx, (cls_name, color) in enumerate(zip(class_names, class_colors)):
            mean = all_means[cls_idx]
            std  = all_stds[cls_idx]
            pdf  = norm.pdf(xs, mean, std)
            ax.plot(xs, pdf, color=color, linewidth=1.8, label=cls_name)
            ax.fill_between(xs, pdf, alpha=0.07, color=color)

        ax.axvline(x_val, color='black', linewidth=2, linestyle='--', zorder=5)

        for cls_idx, (color) in enumerate(class_colors):
            mean = all_means[cls_idx]
            std  = all_stds[cls_idx]
            likelihood = norm.pdf(x_val, mean, std)
            ax.scatter(x_val, likelihood, color=color, s=50, zorder=6,
                       edgecolors='white', linewidths=0.8)

        ax.set_title(feature_names[feat_idx], fontsize=9, fontweight='bold', pad=4)
        ax.set_xlabel("Feature value", fontsize=7, labelpad=2)
        ax.set_ylabel("Probability density", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.15, linewidth=0.5)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [mpatches.Patch(color=class_colors[i], label=class_names[i])
                      for i in range(n_classes)]
    legend_handles.append(plt.Line2D([0], [0], color='black', linewidth=2,
                                     linestyle='--', label='New sample'))
    fig.legend(handles=legend_handles,
               loc='lower center',
               ncol=min(n_classes + 1, 6),
               fontsize=9,
               framealpha=0.3,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Naive Bayes: Learned Gaussian Distributions per Feature\n"
                 "(dashed line = new sample, dots = likelihood at sample point)",
                 fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.show()


def plot_posterior(model: GaussianNB,
                   sample: np.ndarray,
                   class_names: list = None):

    sample = np.asarray(sample).ravel().reshape(1, -1)
    probs = model.predict_proba(sample)[0]
    predicted_idx = np.argmax(probs)

    if class_names is None:
        class_names = [str(c) for c in model.classes_]

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(class_names))]
    edge_colors = ['gold' if i == predicted_idx else 'none'
                   for i in range(len(class_names))]

    order = np.argsort(probs)[::-1]
    sorted_names  = [class_names[i] for i in order]
    sorted_probs  = probs[order]
    sorted_colors = [colors[i] for i in order]
    sorted_edges  = [edge_colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(sorted_names, sorted_probs * 100,
                  color=sorted_colors, edgecolor=sorted_edges,
                  linewidth=2, zorder=3)

    for bar, p in zip(bars, sorted_probs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{p*100:.1f}%",
                ha='center', va='bottom', fontsize=9, color='white')

    ax.set_ylabel("Posterior Probability (%)", fontsize=10)
    ax.set_title(f"Naive Bayes Posterior — Predicted: {class_names[predicted_idx]}  "
                 f"({probs[predicted_idx]*100:.1f}%)",
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(sorted_probs) * 100 * 1.2)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    plt.tight_layout()
    plt.show()

def plot_loglikelihood_heatmap(model,
                               sample: np.ndarray,
                               feature_names: list,
                               class_names: list = None,
                               max_features: int = 20):

    from scipy.stats import norm as _norm

    sample = np.asarray(sample).ravel()
    n_features = min(len(feature_names), max_features)
    feature_names = feature_names[:n_features]
    sample = sample[:n_features]

    if class_names is None:
        class_names = [str(c) for c in model.classes_]

    n_classes = len(class_names)

    # Build log-likelihood matrix shape: (n_features, n_classes)
    ll_matrix = np.zeros((n_features, n_classes))
    for f in range(n_features):
        for c in range(n_classes):
            mean = model.theta_[c, f]
            std  = np.sqrt(model.var_[c, f])
            ll_matrix[f, c] = _norm.logpdf(sample[f], mean, std)

    # --- NEW: compute total log-likelihood per class (sum down each column) ---
    col_totals = ll_matrix.sum(axis=0)  # shape (n_classes,)

    # Normalise totals to probabilities using softmax so they sum to 100%
    shifted = col_totals - col_totals.max()  # shift for numerical stability
    exp_vals = np.exp(shifted)
    probabilities = exp_vals / exp_vals.sum()  # shape (n_classes,)
    predicted_idx = np.argmax(probabilities)

    # --- Build extended matrix: original rows + a separator + totals row ---
    separator_row = np.full((1, n_classes), np.nan)   # blank divider row
    totals_row    = probabilities.reshape(1, n_classes)

    extended_matrix = np.vstack([ll_matrix, separator_row, totals_row])

    # Row labels: feature names + separator + totals label
    extended_row_labels = feature_names + ["", "Posterior %"]

    # --- Plot ---
    n_rows_display = n_features + 2  # features + separator + totals
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.2),
                                    max(6, n_rows_display * 0.45)))

    # Mask the separator row so it shows as blank
    import matplotlib.colors as mcolors
    masked = np.ma.masked_invalid(extended_matrix)

    # Use two separate images: main heatmap and totals row with different cmaps
    # Draw main heatmap (all rows except totals)
    main_matrix = np.ma.masked_invalid(extended_matrix[:-1, :])
    im_main = ax.imshow(main_matrix, aspect='auto', cmap='RdYlGn',
                        extent=[-0.5, n_classes - 0.5, n_rows_display - 0.5, 0.5])

    # Draw totals row with a separate colormap (blues)
    totals_display = totals_row * 100  # convert to percentage for display
    im_totals = ax.imshow(totals_display, aspect='auto', cmap='Blues',
                          extent=[-0.5, n_classes - 0.5, n_rows_display + 0.5, n_rows_display - 0.5],
                          vmin=0, vmax=100)

    plt.colorbar(im_main,   ax=ax, label="Log-likelihood  log P(xⱼ | class)", pad=0.01, fraction=0.03)
    plt.colorbar(im_totals, ax=ax, label="Posterior Probability (%)", pad=0.06, fraction=0.03)

    # Axis labels
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(n_rows_display))
    ax.set_yticklabels(extended_row_labels, fontsize=8)

    ax.set_title("Log-Likelihood Heatmap + Posterior Probability per Class\n"
                 "Green = good fit, Red = poor fit  |  Bottom row = final class probability",
                 fontsize=11, fontweight='bold')

    # Annotate feature cells
    for f in range(n_features):
        for c in range(n_classes):
            ax.text(c, f + 1, f"{ll_matrix[f, c]:.1f}",
                    ha='center', va='center', fontsize=6.5,
                    color='black' if ll_matrix[f, c] > ll_matrix.mean() else 'white')

    # Annotate totals row — highlight predicted class in gold
    for c in range(n_classes):
        prob_pct = probabilities[c] * 100
        ax.text(c, n_rows_display, f"{prob_pct:.1f}%",
                ha='center', va='center', fontsize=8,
                fontweight='bold',
                color='gold' if c == predicted_idx else 'white')

    # Draw a horizontal line to visually separate the totals row
    ax.axhline(y=n_rows_display - 0.5, color='white', linewidth=2, linestyle='--')

    plt.tight_layout()
    plt.savefig("nb_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPredicted class: {class_names[predicted_idx]} ({probabilities[predicted_idx]*100:.1f}%)")

def visualize_naive_bayes(model: GaussianNB,
                          x_test: np.ndarray,
                          y_test: np.ndarray,
                          feature_names: list,
                          class_names: list = None,
                          sample_index: int = 0,
                          max_features: int = 20):

    sample = x_test[sample_index]
    true_label = y_test[sample_index]
    predicted_label = model.predict(sample.reshape(1, -1))[0]

    if class_names is None:
        class_names = [str(c) for c in model.classes_]

    print("=" * 55)
    print(f"  Sample index : {sample_index}")
    print(f"  True label   : {true_label}")
    print(f"  Predicted    : {predicted_label}")
    print(f"  Correct      : {'✓' if true_label == predicted_label else '✗'}")
    print("=" * 55)

    plot_nb_gaussians(model, feature_names, sample,
                      class_names=class_names,
                      max_features=max_features)

    plot_posterior(model, sample, class_names=class_names)

    plot_loglikelihood_heatmap(model, sample, feature_names,
                               class_names=class_names,
                               max_features=max_features)
