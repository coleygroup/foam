import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

from plotting_style import PALETTE, HEXPLOT_CMAP, styleset
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


styleset()

## Figure 2: View timesteps over trajectories

def annotate_argmax(ax, x, y, color='k', step_type='generations', label=None, xytext=(8, 4), log=False, m='o'):
    """
    For a given subplot, plot the argmax point on the curve, with the corresponding label if provided. 
    
    Args:
        ax: The subplot to plot on.
        x: The x values of the curve.
        y: The y values of the curve.
        color: The color of the point.
        step_type: The type of step (generations or calls).
        label: The label to plot.
        xytext: The text to plot.
        log: Whether to use log scale.
        m: The marker to use.
    """
    idx = np.nanargmax(y)
    xmax, ymax = x[idx], y[idx]
    if m:
        
        ax.scatter([xmax], [ymax], s=8, color=color, zorder=10)
    if log:
        xmax_true = xmax - 1 # need to substract off. 

    if "Best" in label or "Top" in label:
        ax.annotate(
            f"{ymax:.2f}",
            #f"{label or ''} Maximum: \n{ymax * 100:.1f}% at {int(xmax_true)} {step_type}",
            xy=(xmax - 1, ymax),
            xytext=xytext,
            textcoords='offset points',
            ha='left',
            va='bottom',
            fontsize=6.5,
        )
    elif "encounter" in label:
        ax.annotate(
            f"{ymax:.2f}\nencountered",
            xy=(35, 0.720),
            xytext=xytext,
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=6.5,
        )
    elif "Improvement" in label:
        ax.scatter([xmax], [ymax], s=8, color=color, zorder=10)
        idxmin = np.nanargmin(y)
        ymin = y[idxmin]
        improvement = ymax - ymin
        ax.annotate(
            f"{label or ''}:\n+{improvement:.2f}",
            xy=(23, 0.5),
            xytext=(23, 0.5),
            textcoords='offset points',
            ha='right',
            va='bottom',
            fontsize=6.5,
        )
    return xmax, ymax

def make_plot(plot_data, step_type='generation', log=False, 
              error_bars=True, ci=True, m='', 
              auto_annotate=True, top10=None, shaping='rectangle'):
    """
    Make a plot of the metrics over time.
    
    Args:
        plot_data: The data to plot.
        step_type: The type of step (generations or calls).
        log: Whether to use log scale.
        error_bars: Whether to plot error bars.
        ci: Whether to plot confidence intervals.
        m: The marker to use.
        auto_annotate: Whether to automatically annotate the plot.
        top10: Whether to plot top 10 metrics.
        shaping: The shaping of the plot (rectangle or square).
    """
    plot_data = plot_data.copy()
    if step_type == 'calls':
        step_col = "Meta.Calls Made"
        range_vals = [0, 600] + list(range(750, 7501, 250))
    elif step_type == "generation":
        step_col = "generation"
        range_vals = range(0, min(60, max(plot_data[step_col])))
        plot_data = plot_data.loc[range_vals, :]

    if shaping=='rectangle':
        fig, axes = plt.subplots(4, 1, figsize=(3,8.5))
        plt.subplots_adjust(hspace=0.1)
    else: 
        fig, axes = plt.subplots(2, 2, figsize=(7, 5))
        axes = axes.flatten()
    
    for ax in axes.flatten():
        for spine in ax.spines.values():
            spine.set_color('k')      # black
            spine.set_linewidth(1)    # optional: make lines thicker
        ax.set_ylim(0, 1)
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.set_xlim(0, list(range_vals)[-1])
    

    
    if "top10_avg_entropy_sim" in plot_data.columns:
        ax0_ylabel = "Average of Top 10 \n spectral similarities observed"
        ax0_column = "top10_avg_entropy_sim"
    else:
        ax0_ylabel = "Best spectral \n similarity observed"
        ax0_column = "top1_entropy"

    axes[0].plot(plot_data[step_type], plot_data[ax0_column], marker=m, color=PALETTE[0], label='Spectral Similarity', linewidth=1)
    axes[0].set_ylabel(ax0_ylabel)
    axes[0].legend()
    if auto_annotate:
        annotate_argmax(axes[0], plot_data[step_type], plot_data[ax0_column], 
                        color=PALETTE[0], label='Improvement\nobserved', m=m, step_type=step_type, log=log)
        
    axes[1].plot(plot_data[step_type], plot_data['inchikey_match'] , marker=m, color=PALETTE[1], label='Encounter Rate', linewidth=1)
    axes[1].set_ylabel("Encounter Rate")
    axes[1].legend()
    if auto_annotate:
        annotate_argmax(axes[1], plot_data[step_type], plot_data['inchikey_match'], 
                        color=PALETTE[1], label='Final encounter rate', step_type=step_type, log=log)
        
    # two options: if dashed, use two colors, palette[2] for top1tani, palette[3] for top1match, dashed for top10 xx. 
    
    axes[2].plot(plot_data[step_type], plot_data['top1_tani'], marker=m, color=PALETTE[2], label=f'Best of 1', linewidth=1)
    if top10== 'dashed':
        axes[2].plot(plot_data[step_type], plot_data['top10_max_tani'], 
                     marker=m, ls='--',
                     color=PALETTE[2], label=f'Best of 10', linewidth=1)
    else:
        axes[2].plot(plot_data[step_type], plot_data['top10_max_tani'], marker=m, 
                     color=PALETTE[3], label=f'Best of 10', linewidth=1)
    
    if auto_annotate:
        annotate_argmax(axes[2], plot_data[step_type], plot_data['top1_tani'], 
                        color=PALETTE[2], label='Top 1', step_type=step_type, log=log)
        
        annotate_argmax(axes[2], plot_data[step_type], plot_data['top10_max_tani'], 
                        color=PALETTE[3] if top10 != 'dashed' else PALETTE[2], label='Best of 10', step_type=step_type,log=log)
    axes[2].set_ylabel("Structural Similarity \n (Tanimoto, 2048-bit Morgan FP)")
    axes[2].legend()

    axes[3].plot(plot_data[step_type], plot_data['top1_match'], marker=m, color=PALETTE[4], label=f'Top 1 Exact Match')
    if top10 == 'dashed':
        axes[3].plot(plot_data[step_type], plot_data['top10_match'], 
                     marker=m, ls='--',
                     color=PALETTE[4], label=f'Top 10 Exact Match')
        
    else:
        axes[3].plot(plot_data[step_type], plot_data['top10_match'], marker=m, color=PALETTE[5], label=f'Top 10 Exact Match')
    

    axes[3].plot(plot_data[step_type], plot_data['inchikey_match'],
                 marker=m, alpha=0.5, ls="dotted",
                 color=PALETTE[3], label=f'Encounter Rate')

    if auto_annotate:
        annotate_argmax(axes[3], plot_data[step_type], plot_data['top1_match'], 
                        color=PALETTE[4], label='Top 1', step_type=step_type, log=log)
        annotate_argmax(axes[3], plot_data[step_type], plot_data['top10_match'], 
                        color=PALETTE[5] if top10 != 'dashed' else PALETTE[4], label='Top 10', step_type=step_type, log=log)

    axes[3].set_ylabel("Exact Match (Proportion)")
    axes[3].legend()

    axes[3].set_xlabel(step_type.capitalize())
    if shaping=='square':
        axes[2].set_xlabel(step_type.capitalize())

    if log:
        for ax in axes.flatten():
            #ax.set_xscale('asinh')
            ax.set_xscale('symlog', linscale=0.5)
            ticks = [0,1,2,3,4,5,6,7,8,9] + [10,20,30,40,50,60]
            tick_labels = ["0", "$10^0$"] + [None] * 8 + [ "$10^1$"] + [None] * 4 + [r"$6\times10^1$"]
            ax.set_xticks(ticks, labels=tick_labels)

            ax.set_xlabel(f'{step_type.capitalize()}')
    
    if error_bars:
        for metric, idx, pal in zip([ax0_column, 'inchikey_match', 'top1_tani', 'top10_max_tani', 'top1_match', 'top10_match'],
                                     [0, 1, 2, 2, 3, 3],
                                    [0, 1, 2, 2, 4, 4]):
            axes[idx].errorbar(
                plot_data[step_type],
                plot_data[metric],
                yerr=plot_data[f"{metric}_sem"],
                fmt='none', #m,
                color=PALETTE[pal],
                alpha=1, 
                linewidth=1,
                
            )
    if ci:
        for i, metric in enumerate([ax0_column, 'inchikey_match', 'top1_tani']):
            axes[i].fill_between(plot_data[step_type],
                                plot_data[f"{metric}_ci_lower"],
                                plot_data[f"{metric}_ci_upper"],
                                color=PALETTE[i],
                                alpha=0.3)
        
        axes[2].fill_between(plot_data[step_type],
                                plot_data[f"{'top10_max_tani'}_ci_lower"],
                                plot_data[f"{'top10_max_tani'}_ci_upper"],
                                color=PALETTE[2] if top10=='dashed' else PALETTE[3],
                                alpha=0.2)

        axes[3].fill_between(plot_data[step_type],
                                plot_data[f"{'top1_match'}_ci_lower"],
                                plot_data[f"{'top1_match'}_ci_upper"],
                                color=PALETTE[4],
                                alpha=0.2)
        axes[3].fill_between(plot_data[step_type],
                                plot_data[f"{'top10_match'}_ci_lower"],
                                plot_data[f"{'top10_match'}_ci_upper"],
                                color=PALETTE[4] if top10=='dashed' else PALETTE[5],
                                alpha=0.2)

    plt.tight_layout()
    plt.show()
    return fig, axes
    

def fig2_from_timestep_df(timestep_df, log=False, error_bars=True, ci=True, m=''):
    if ci and not any([col for col in timestep_df.columns if "_upper" in col]):
        raise ValueError("CI requested but no CI columns found in dataframe")
    
    fig, axes = make_plot(timestep_df, log=log, error_bars=error_bars, ci=ci, m=m)
    return fig, axes




def fig3a_hexplot(summaries, col="top1_tani@3", fig=False, color_label="Top 1 Structural Similarity", fig_suffix=''):
    norm = mcolors.Normalize(vmin=-0.2, vmax=1.0)
    hexplot = sns.jointplot(x=summaries["NDSBestMol.target_self_entropy"], 
                            y=summaries["best_seed_sim"], C=summaries[col], kind="hex", 
                            cmap=HEXPLOT_CMAP, #'Greens', 
                            gridsize=30, 
                            norm=norm, 
                            joint_kws={'edgecolor': 'white', 'linewidth': 0.0000001},
                            height=5) # PowerNorm(gamma=0.60))
    hexplot.ax_marg_x.remove()
    hexplot.ax_marg_y.remove()

    hexplot.ax_joint.tick_params(
    axis='y',          # Apply to the y-axis
    left=True,         # Keep left ticks on
    right=False,        # Turn right ticks on
    labelleft=True,    # Keep left labels on
    labelright=False    # Turn right labels on
    )
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # hexplot.fig.set_figwidth(6)
    # hexplot.fig.set_figheight(4)
    # hexplot.ax_joint.yaxis.tick_right()
    # hexplot.ax_joint.yaxis.set_label_position("right")
    # hexplot.ax_joint.spines['left'].set_visible(False)
    #hexplot.ax_joint.spines['right'].set_visible(True)
    hexplot.ax_joint.invert_yaxis()

    plt.ylabel('Closest seed\'s structural similarity \n (Tanimoto, 2048-bit Morgan)')
    plt.xlabel('Spectral similarity for true structure')
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
    # make new ax object for the cbar

    # mappable = ax.collections[0]

    # --- ADD/MODIFY THIS SECTION ---
    # Define the boundaries and ticks you want to show
    cbar_boundaries = np.linspace(0, 1, 100)  # Creates [0, 0.1, 0.2, ..., 1.0]
    cbar_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    cbar_ax = hexplot.fig.add_axes([0.2, 0.77, 0.50, 0.03])
    # 8. Pass boundaries and ticks to the colorbar
    cbar = plt.colorbar(
        # mappable,
        cax=cbar_ax,
        label=color_label,
        orientation="horizontal",
        boundaries=cbar_boundaries,  # <-- Truncates the bar
        ticks=cbar_ticks             # <-- Sets the ticks
    )
    
    if fig:
        plt.savefig(f"../media/hexplot_{fig_suffix}.pdf", transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

from plotting_style import GRAY_CMAP, PINK_CMAP



def fig3b_plot_seed_kde(summaries, cols_to_plot: list, success_col, k, fig=False, fig_suffix=''):
    print(summaries['best_seed_bin'].unique())
    valid_bins = sorted(summaries['best_seed_bin'].unique())
    print(valid_bins)
    PINK_CMAP_DICT = dict(zip(valid_bins, PINK_CMAP))

    def plot_seed_kde_helper(col_name, k, color, label, **kwargs):
        plt.gca().set_facecolor("none")
        # Pick the right color map based on that column
        #cmap = CUBEHELIX_CMAP.get(col_name, None) # TODO: may need to update dict to take in a k as well, if we wanted multiple improvements over time??
        cmap = PINK_CMAP_DICT
        if cmap is None:
            raise ValueError(f"No palette defined for column '{col_name}'")

        if 'step0' not in col_name: 
            sns.kdeplot(
                x=kwargs["data"][f'{col_name}@{k}'],
                fill=True, alpha=0.5, linewidth=1.5, bw_adjust=0.5,
                clip=(0,1),
                color=cmap[label]
            )
        else:
            sns.kdeplot(
                x=kwargs["data"][f'{col_name}'],
                fill=True, alpha=0.5, linewidth=1.5, bw_adjust=0.5,
                clip=(0,1),
                color=cmap[label]
            )

   
    g = sns.FacetGrid(
        summaries, 
        row="best_seed_bin", 
        hue="best_seed_bin",
        aspect=4.5, 
        height=0.68,
        row_order=sorted(valid_bins),
        gridspec_kws={"hspace":-0.3}
    )
    g.set(ylim=(0, 12))
    ### background block plotting. 

    def add_background(ax, label):
        color = GRAY_CMAP[valid_bins.index(label)]
        low, high = label.left, label.right
        ax.axvspan(low, high, ymin=0, ymax=0.5, color=color, alpha=0.3, zorder=0, ls='--')

    for i, val in enumerate(sorted(valid_bins)):
        ax = g.axes[i, 0] if g.axes.ndim > 1 else g.axes[i]
        add_background(ax, val)
        
    ### plotting KDEs 
    for col in cols_to_plot:
        g.map_dataframe(plot_seed_kde_helper, col, k=k, label="best_seed_bin") # Change to pull from starting population.

    ### annotating bins with counts 
    counts = summaries["best_seed_bin"].value_counts().to_dict()
    # masses = summaries.groupby('best_seed_bin', observed=False).agg({'mass': 'mean'})
    # masses.index = masses.index.map(str)
    # # turn counts keys to string
    counts = {str(k): v for k, v in counts.items()}
    
    def label_seed_bin(x, color, label):
        ax = plt.gca()
        count = counts.get(label, 0)
        # mass = masses.loc[label, 'mass']
        color = 'k' #list(bg_palette.values())[4] # flatten to just be the third color lol 
        ax.text(-0.02, 0.25, f"{label} \n (n={count}) \n", #+ "$\\overline{m/z}:$" +  f" {mass:.2f}", 
                color=color,
                ha="right", va="center", transform=ax.transAxes)

    g.map(label_seed_bin, f"{cols_to_plot[0]}@{k}")

    summaries['success'] = (summaries[f'{success_col}@{k}'] == 1).astype(int)
    #bin_success = summaries.groupby('max_seed_bin').agg({'best_top10_max_tani=1': ['mean', 'sum'], 'best_top10_avg_tani': ['mean', 'std']})
    bin_success = summaries.groupby('best_seed_bin', observed=False).agg({'success': ['mean', 'sum']})
    bin_success.index = bin_success.index.map(str)
    def label_success(x, color, label):
        PINK_CMAP_DICT = dict(zip(valid_bins, PINK_CMAP))
        ax = plt.gca()
        success = bin_success.loc[label, ('success', 'sum')]
        percent = bin_success.loc[label, ('success', 'mean')] * 100
        try:
            color = PINK_CMAP_DICT[label]
        except:
            PINK_CMAP_DICT = {str(k): v for k, v in PINK_CMAP_DICT.items()}
            color = PINK_CMAP_DICT[label]
        ax.text(1.03, .3, f"{percent:.1f}% \n(k={success})",  #  fontweight="bold", 
                color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label_success, f"{success_col}@{k}")

    top_ax = g.axes[0, 0] 

    # Header for the left side ("Seed bin")
    top_ax.annotate(
        "Similarity bin",
        xy=(-0.02, 0.84),            # (x, y) position in axes fraction
        xycoords='axes fraction',
        ha="right",                 # Horizontal alignment
        va="center",
        fontsize=7,
        # fontweight="bold",
        color="k"
    )

    # Header for the right side ("Top 10 match")
    top_ax.annotate(
        "Match in\nTop 10",
        xy=(1.03, 0.84),             # (x, y) position in axes fraction
        xycoords='axes fraction',
        ha="left",                 # Horizontal alignment
        va="center",
        fontsize=7,
        #fontweight="bold",
        color='k'
    )

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xlim=(0, 1))
    g.despine(left=True)
        
    # print(palette_dict[cols_to_plot[0]])
    col_maps = {f"{cols_to_plot[0]}": f"Best of Top 10 after {k} gens."}
    legend_patches = [mpatches.Patch(color=PINK_CMAP[0], alpha=0.9, label=col_maps[col]) for col in cols_to_plot]
    legend_patches += [mpatches.Patch(color=GRAY_CMAP[0], alpha=0.7, ls = "--", label='Seed similarity bin')]
    g.figure.legend(
        handles=legend_patches,
        loc='upper right', bbox_to_anchor=(0.9, 0.87),
        # title="Metric",
        #frameon=True,
    )
    g.set_axis_labels("Best Structural Similarity", "")

    return fig, g



def fig3c_plot_oracle_kde(summaries, cols_to_plot: list, success_col, k, fig=False, fig_suffix=''):
    valid_bins = sorted(summaries['oracle_accuracy_bin'].unique())
    PINK_CMAP_DICT = dict(zip(valid_bins, PINK_CMAP))

    def plot_kde(col_name, k, color, label, **kwargs):
        plt.gca().set_facecolor("none")
        cmap = PINK_CMAP_DICT
        if cmap is None:
            raise ValueError(f"No palette defined for column '{col_name}'")

        if 'step0' not in col_name:
            sns.kdeplot(
                y=kwargs["data"][f'{col_name}@{k}'],
                fill=True, alpha=0.5, linewidth=1.5, bw_adjust=0.5,
                clip=(0,1),
                color=cmap[label]
            )
        else:
            sns.kdeplot(
                y=kwargs["data"][f'{col_name}'],
                fill=True, alpha=0.5, linewidth=1.5, bw_adjust=0.5,
                clip=(0,1),
                color=cmap[label]
            )

    g = sns.FacetGrid(
        summaries,
        col="oracle_accuracy_bin",
        hue="oracle_accuracy_bin",
        aspect=0.2,
        height=3,
        col_order=sorted(valid_bins),
        gridspec_kws={"wspace":-0.1}
    )
    
    for col in cols_to_plot:
        g.map_dataframe(plot_kde, col, k=k, label="oracle_accuracy_bin")

    ### annotating bins with counts
    counts = summaries["oracle_accuracy_bin"].value_counts().to_dict()
    counts = {str(k): v for k, v in counts.items()}

    def label_oracle_bin(x, color, label):
        ax = plt.gca()
        print(label)
        count = counts.get(label, 0)
        color = 'k'
        ax.text(0.4, 1.00, f"{label}\n(n={count}) \n",
                color=color,
                rotation=30,
                ha="center", va="bottom", transform=ax.transAxes)

    g.map(label_oracle_bin, f"{cols_to_plot[0]}@{k}")

    top_ax = g.axes[0, 0]
    top_ax.annotate(
        "Accuracy\nbin",
        xy=(-0.6, 1.1),
        xycoords='axes fraction',
        ha="center",
        va="bottom",
        fontsize=7,
        color="#333333"
    )

    summaries['success'] = (summaries[f'{success_col}@{k}'] == 1).astype(int)
    bin_success = summaries.groupby('oracle_accuracy_bin', observed=False).agg({'success': ['mean', 'sum']})
    bin_success.index = bin_success.index.map(str)

    def label_success(x, color, label):
        ax = plt.gca()
        success = bin_success.loc[label, ('success', 'sum')]
        percent = bin_success.loc[label, ('success', 'mean')] * 100
        try:
            color = PINK_CMAP_DICT[label]
        except:
            PINK_CMAP_DICT_str = {str(k): v for k, v in PINK_CMAP_DICT.items()}
            color = PINK_CMAP_DICT_str[label]
        ax.text(0.4, -0.05, f"{percent:.1f}% \n(k={success})",
                color=color,
                ha="center", va="top", transform=ax.transAxes)

    g.map(label_success, f"{success_col}@{k}")

    top_ax.annotate(
        "Match in\nTop 10",
        xy=(-1, -0.05),
        xycoords='axes fraction',
        ha="left",
        va="top",
        fontsize=7,
        color="k",
    )

    g.set_titles("")
    g.set(xticks=[], xlabel="")
    g.despine(bottom=True, left=False)
    g.set_axis_labels("", "Best Structural Similarity")
    g.set(ylim=(1.0, 0.0))

    col_maps = {f"{cols_to_plot[0]}": f"Best of Top 10 after {k} gens."}
    legend_patches = [mpatches.Patch(color=PINK_CMAP[0], alpha=0.9, label=col_maps[col]) for col in cols_to_plot]
    legend_patches += [mpatches.Patch(color=GRAY_CMAP[0], alpha=0.7, ls="--", label='Oracle accuracy bin')]
    ax_right = g.axes.flat[-1]
    ax_right.legend(
        handles=legend_patches,
        loc='upper right', bbox_to_anchor=(3, 1),
    )

    g.figure.subplots_adjust(bottom=0.15, top=0.8)

    if fig:
        plt.savefig(f'../media/perf_by_oracle_{fig_suffix}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    plt.show()

    return g

from regressor import measure_logistic_predictor, plot_roc_curve

def fig3d_plot_roc_curve(steps, summaries):
    sim_cols = ['top1_entropy', 
                'generation', 
                'num_seeds',  
                'num_colli_engs',
                'SA_of_best_entropy', 
           ]

    success_col = 'top10_match' 
    for s in sim_cols + ['adduct']:
        if s not in steps.columns and s in summaries.columns:
            # map using the run_name.
            steps = steps.merge(summaries[['run_name', s]], on='run_name', how='left')

    X = steps.groupby('generation').tail(1)[sim_cols].to_numpy(float)
    y = steps.groupby('generation').tail(1)[success_col].to_numpy(float)
    tprs, mean_fpr, metrics, coef_list, final_feature_names = measure_logistic_predictor(X, y, adducts=steps.groupby('generation').tail(1)["adduct"], feature_names=sim_cols)

    
    fig, coef_df = plot_roc_curve(tprs, mean_fpr, metrics, coef_list, final_feature_names, PALETTE)
    return fig, coef_df

    #fig_pred.savefig('../media/fig4d_logistic_predictor_top10match.pdf', dpi=300, transparent=True, bbox_inches='tight')
    
