"""plotting_utils.py

Utilities for interactive visualization of mass spectra using Plotly.
Supports mirror plots for comparing predicted vs experimental spectra,
multi-energy displays, and candidate ranking visualizations.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go


def plot_spectrum(
    spec_dict: Dict[str, np.ndarray],
    second_spec: Optional[Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]] = None,
    energy_key: Optional[str] = None,
    annot_threshold: float = 0.02,
    match_tolerance: float = 0.05,
    candidate_rank: int = 1,
    ent_score: Optional[List[float]] = None,
    sa_score: Optional[List[float]] = None,
    annot_peaks: bool = False,
    second_smi: Optional[List[str]] = None,
) -> go.Figure:
    """Create an interactive mirror plot comparing one or more spectra.

    Generates a Plotly figure with the primary spectrum shown above the x-axis
    and optional secondary spectrum/spectra shown below (mirrored). Includes
    interactive dropdowns for energy selection and candidate ranking.

    Args:
        spec_dict: Primary spectrum as dict mapping collision energy (str) to
            2D numpy array of shape (n_peaks, 2) with [m/z, intensity] columns.
        second_spec: Optional secondary spectrum(s) to compare against. Can be
            a single dict (same format as spec_dict) or a list of dicts for
            multiple candidates.
        energy_key: Initial collision energy to display. If None, uses lowest.
        annot_threshold: Minimum intensity for peak annotation labels.
        match_tolerance: m/z tolerance for determining matched peaks (Da).
        candidate_rank: Initial candidate to display (1-indexed) when multiple
            secondary spectra are provided.
        ent_score: List of entropy similarity scores for each candidate.
        sa_score: List of SA scores for each candidate.
        annot_peaks: If True, annotate peaks with m/z values.
        second_smi: List of SMILES strings for each candidate (for labels).

    Returns:
        Plotly Figure with interactive energy and candidate selection.
    """
    # --------------------------------------------------------
    # (a) Sort energies so that the smallest energy comes first.
    energy_keys = sorted(list(spec_dict.keys()), key=lambda x: float(x))
    if energy_key is None or energy_key not in energy_keys: # pick the first one by default. 
        energy_key = energy_keys[0]

    # --------------------------------------------------------
    # Process second_spec: allow it to be a list or a dict.
    # --------------------------------------------------------
    if second_spec is not None:
        if isinstance(second_spec, list):
            second_specs_list = second_spec
        else:
            second_specs_list = [second_spec]
    else:
        second_specs_list = []
    # Flag indicating whether we have multiple candidate second spectra.
    candidate_dropdown = len(second_specs_list) > 1  
    # “Effective” number of candidate groups.
    num_candidates_effective = len(second_specs_list) if len(second_specs_list) > 0 else 0

    # --------------------------------------------------------
    # Helper functions to build traces.
    # --------------------------------------------------------
    def build_line_trace(mz_arr, intensity_arr, color):
        x_lines = []
        y_lines = []
        for i in range(len(mz_arr)):
            x_lines += [mz_arr[i], mz_arr[i], None]
            y_lines += [0, intensity_arr[i], None]
        return go.Scatter(
            x=x_lines,
            y=y_lines,
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        )

    def build_marker_trace(mz_arr, intensity_arr, texts, color, use_text):
        if use_text:
            return go.Scatter(
                x=mz_arr,
                y=intensity_arr,
                mode='markers+text',
                marker=dict(color=color, size=8),
                text=texts,
                textposition='top center',
                showlegend=False
            )
        else:
            return go.Scatter(
                x=mz_arr,
                y=intensity_arr,
                mode='markers',
                marker=dict(color=color, size=8),
                showlegend=False
            )

    # --------------------------------------------------------
    # Build traces per energy key.
    # --------------------------------------------------------
    energy_traces_primary = {}
    energy_traces_second = {}  # either a single candidate group (if one candidate) or a list of candidate groups

    for key in energy_keys:
        arr1 = spec_dict[key]
        mz1 = arr1[:, 0]
        intensity1 = arr1[:, 1]
        use_text = annot_peaks
        if use_text:
            texts1 = np.where(intensity1 >= annot_threshold, np.round(mz1, 4).astype(str), "")
        else:
            texts1 = None
        # For matching, if a second spectrum is provided, use the first candidate for computing primary matching.
        if len(second_specs_list) > 0 and key in second_specs_list[0]:
            arr2 = second_specs_list[0][key] # Takes the first entry in second_specs_list. 
            mz2 = arr2[:, 0]
            matched_mask_primary = np.array([np.any(np.abs(mz2 - x) < match_tolerance) for x in mz1])
        else:
            matched_mask_primary = np.full(mz1.shape, False)
        # Split primary spectrum peaks.
        mz1_unmatched = mz1[~matched_mask_primary]
        intensity1_unmatched = intensity1[~matched_mask_primary]
        if use_text:
            texts1_unmatched = texts1[~matched_mask_primary]
            texts1_matched = texts1[matched_mask_primary]
        else: 
            texts1_unmatched = None
            texts1_matched = None
        mz1_matched = mz1[matched_mask_primary]
        intensity1_matched = intensity1[matched_mask_primary]

        primary_unmatched_line = build_line_trace(mz1_unmatched, intensity1_unmatched, color='blue')
        primary_unmatched_marker = build_marker_trace(mz1_unmatched, intensity1_unmatched, texts1_unmatched, color='blue', use_text=use_text)
        primary_matched_line = build_line_trace(mz1_matched, intensity1_matched, color='blue')
        primary_matched_marker = build_marker_trace(mz1_matched, intensity1_matched, texts1_matched, color='blue', use_text=use_text)
        energy_traces_primary[key] = [
            primary_unmatched_line, primary_unmatched_marker,
            primary_matched_line, primary_matched_marker
        ]

        # Build second spectrum traces.
        if num_candidates_effective == 0:
            energy_traces_second[key] = []
        elif not candidate_dropdown:
            # Single candidate—behave as before.
            candidate = second_specs_list[0]
            if key in candidate and len(arr2[:, 0]) > 0:
                arr2 = candidate[key]
                mz2 = arr2[:, 0]
                intensity2 = -1 * arr2[:, 1]
                use_text = annot_peaks
                if use_text:
                    texts2 = np.where(np.abs(intensity2) >= annot_threshold, np.round(mz2, 4).astype(str), "")
                else:
                    texts2 = None
                matched_mask = np.array([np.any(np.abs(mz1 - m) < match_tolerance) for m in mz2])
                mz2_matched = mz2[matched_mask]
                intensity2_matched = intensity2[matched_mask]
                if use_text:
                    texts2_matched = texts2[matched_mask]
                    texts2_unmatched = texts2[~matched_mask]
                else:
                    texts2_matched = None
                    texts2_unmatched = None
                mz2_unmatched = mz2[~matched_mask]
                intensity2_unmatched = intensity2[~matched_mask]
                

                second_unmatched_line = build_line_trace(mz2_unmatched, intensity2_unmatched, color='red')
                second_unmatched_marker = build_marker_trace(mz2_unmatched, intensity2_unmatched, texts2_unmatched, color='red', use_text=use_text)
                second_matched_line = build_line_trace(mz2_matched, intensity2_matched, color='red')
                second_matched_marker = build_marker_trace(mz2_matched, intensity2_matched, texts2_matched, color='red', use_text=use_text)
                group = [second_unmatched_line, second_unmatched_marker,
                         second_matched_line, second_matched_marker]
            else:
                group = [
                    go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                    go.Scatter(x=[], y=[], mode='markers+text', showlegend=False),
                    go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                    go.Scatter(x=[], y=[], mode='markers+text', showlegend=False)
                ]
            energy_traces_second[key] = group
        else:
            # Multiple candidate groups: one group per candidate.
            groups = []
            for candidate in second_specs_list:
                if key in candidate:
                    arr2 = candidate[key]
                    mz2 = arr2[:, 0]
                    intensity2 = -1 * arr2[:, 1]

                    # Handle empty mz2
                    if mz2.size == 0:
                        group = [
                            go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                            go.Scatter(x=[], y=[], mode='markers+text', showlegend=False),
                            go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                            go.Scatter(x=[], y=[], mode='markers+text', showlegend=False)
                        ]
                    else:
                        use_text = annot_peaks
                        if use_text:
                            texts2 = np.where(np.abs(intensity2) >= annot_threshold, np.round(mz2, 4).astype(str), "")
                        else:
                            texts2 = None
                        matched_mask = np.array([np.any(np.abs(mz1 - m) < match_tolerance) for m in mz2])
                        mz2_matched = mz2[matched_mask]
                        intensity2_matched = intensity2[matched_mask]
                        if use_text:
                            texts2_matched = texts2[matched_mask]
                            texts2_unmatched = texts2[~matched_mask]
                        else:
                            texts2_matched = None
                            texts2_unmatched = None
                        mz2_unmatched = mz2[~matched_mask]
                        intensity2_unmatched = intensity2[~matched_mask]

                        cand_unmatched_line = build_line_trace(mz2_unmatched, intensity2_unmatched, color='red')
                        cand_unmatched_marker = build_marker_trace(mz2_unmatched, intensity2_unmatched, texts2_unmatched, color='red', use_text=use_text)
                        cand_matched_line = build_line_trace(mz2_matched, intensity2_matched, color='red')
                        cand_matched_marker = build_marker_trace(mz2_matched, intensity2_matched, texts2_matched, color='red', use_text=use_text)
                        group = [cand_unmatched_line, cand_unmatched_marker,
                                 cand_matched_line, cand_matched_marker]
                else:
                    group = [
                        go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                        go.Scatter(x=[], y=[], mode='markers+text', showlegend=False),
                        go.Scatter(x=[], y=[], mode='lines', showlegend=False),
                        go.Scatter(x=[], y=[], mode='markers+text', showlegend=False)
                    ]
                groups.append(group)
            energy_traces_second[key] = groups

    # --------------------------------------------------------
    # Assemble all traces into one figure.
    # Ordering: primary (4 traces) then candidate traces.
    # --------------------------------------------------------
    fig = go.Figure()
    all_traces = []
    primary_matched_indices = []
    candidate_indices_by_candidate = ([[] for _ in range(num_candidates_effective)] 
                                      if candidate_dropdown and num_candidates_effective > 0 else None)
    candidate_matched_indices = []

    default_energy = energy_keys[0]  # smallest energy

    for key in energy_keys:
        group_primary = energy_traces_primary.get(key, [])
        if num_candidates_effective == 0:
            group_second = []
        elif not candidate_dropdown:
            group_second = energy_traces_second.get(key, [])
        else:
            candidate_groups = energy_traces_second.get(key, [])
            group_second = []
            for candidate_group in candidate_groups:
                group_second.extend(candidate_group)
        group = group_primary + group_second

        # --------------------------------------------------
        # (b) Set default visibility:
        # For the default energy (smallest), use the candidate_rank parameter:
        #   - if candidate_rank is None: show all candidate groups,
        #   - otherwise show only the candidate group corresponding to candidate_rank.
        # For all other energies, hide all traces.
        # --------------------------------------------------
        if key == default_energy:
            for trace in group_primary:
                trace.visible = True
            if num_candidates_effective > 0:
                if candidate_dropdown:
                    desired = 0 if candidate_rank is None else candidate_rank - 1
                    for i in range(num_candidates_effective):
                        for j in range(4):
                            idx = len(group_primary) + i * 4 + j
                            if candidate_rank is None:
                                group[idx].visible = True
                            else:
                                group[idx].visible = (i == desired)
                else:
                    for trace in group_second:
                        trace.visible = True
        else:
            for trace in group:
                trace.visible = False

        base = len(all_traces)
        primary_matched_indices.extend([base + 2, base + 3])
        if num_candidates_effective > 0:
            if not candidate_dropdown:
                second_matched_index_start = base + 4
                candidate_matched_indices.extend([second_matched_index_start + 2, second_matched_index_start + 3])
            else:
                for i in range(num_candidates_effective):
                    idx_start = base + 4 + i * 4
                    candidate_indices_by_candidate[i].extend([idx_start + 2, idx_start + 3])
        all_traces.extend(group)
    for trace in all_traces:
        fig.add_trace(trace)

    # --------------------------------------------------------
    # Build energy selection dropdown.
    # For each energy key, if selected, always show that energy’s primary traces and candidate traces.
    # (For nondefault energies the candidate traces remain hidden.)
    # --------------------------------------------------------
    total_traces = len(all_traces)
    buttons_energy = []
    base = 0
    for key in energy_keys:
        if num_candidates_effective == 0:
            group_size = 4
        elif not candidate_dropdown:
            group_size = 4 + 4
        else:
            group_size = 4 + 4 * num_candidates_effective
        vis = [False] * total_traces
        for j in range(base, base + 4):
            vis[j] = True
        if num_candidates_effective > 0:
            if candidate_dropdown:
                # For nondefault energies, leave candidate traces visible as set (usually all False)
                for j in range(base + 4, base + 4 * (num_candidates_effective + 1)):
                    vis[j] = True
            else:
                for j in range(base + 4, base + 8):
                    vis[j] = True
        # For nondefault energy groups, we clear any annotation.
        score_text = ""
        if (not candidate_dropdown) and ((sa_score is not None) or (ent_score is not None)):
            if (sa_score is not None) and (ent_score is not None):
                score_text = f"SA Score: {sa_score[0]:.2f}, Entropy Score: {ent_score[0]:.2f}"
            elif sa_score is not None:
                score_text = f"SA Score: {sa_score[0]:.2f}"
            elif ent_score is not None:
                score_text = f"Entropy Score: {ent_score[0]:.2f}"
        annotation = dict(
            x=0.5, y=1.08, xref="paper", yref="paper",
            text=score_text,
            showarrow=False,
            font=dict(size=14)
        ) if score_text else {}
        buttons_energy.append(dict(
            label=key,
            method="update",
            args=[{"visible": vis}, {"annotations": [annotation] if score_text else []}]
        ))
        base += group_size
    # "All" option.
    vis_all = []
    for key in energy_keys:
        if num_candidates_effective == 0:
            vis_all.extend([True] * 4)
        elif not candidate_dropdown:
            vis_all.extend([True] * (4 + 4))
        else:
            if candidate_rank is None:
                vis_all.extend([True] * 4 + [True] * (4 * num_candidates_effective))
            else:
                vis_all.extend([True] * 4 + [True] * 4 + [False] * (4 * (num_candidates_effective - 1)))
    buttons_energy.append(dict(
        label="All",
        method="update",
        args=[{"visible": vis_all}, {"annotations": []}]
    ))
    updatemenus = [{
        "active": energy_keys.index(energy_key),
        "buttons": buttons_energy,
        "x": 1.05,  # shift leftward a bit
        "y": 1
    }]

    # --------------------------------------------------------
    # Build candidate (second spectrum) selection dropdown.
    #
    # FIXED: Now respects the currently selected energy instead of hardcoding to default energy.
    # --------------------------------------------------------
    if candidate_dropdown:
        # Compute the indices for the candidate traces in the default energy group.
        default_candidate_indices = list(range(4, 4 + 4 * num_candidates_effective))
        total_traces = len(all_traces)  # full length of the figure
        
        # Build candidate update visible vectors that respect the current energy selection
        candidate_button_options = {}
        # For each candidate option, compute a candidate block (of length 4*num_candidates_effective)
        # For the default energy group, the candidate block will be:
        #   - For option i: candidate traces for candidate group i become True (4 entries)
        #     and all other candidate groups in that energy become False.
        for i in range(num_candidates_effective):
            candidate_button_options[i] = []
            for j in range(num_candidates_effective):
                if j == i:
                    candidate_button_options[i].extend([True]*4)
                else:
                    candidate_button_options[i].extend([False]*4)
        
        # FIXED: Create energy-aware visibility function that respects current energy selection
        def full_visible_vector_for_candidate(candidate_option, current_energy_key=None):
            # Number of energy groups.
            num_groups = len(energy_keys)
            
            # If no current energy specified, use the default (first) energy
            if current_energy_key is None:
                current_energy_key = energy_keys[0]
            
            # Find the index of the current energy
            current_energy_idx = energy_keys.index(current_energy_key)
            
            # Build visibility vector that shows the selected energy with the selected candidate
            full_vec = []
            for group_idx in range(num_groups):
                if group_idx == current_energy_idx:
                    # For the selected energy: primary traces (4 True) + candidate block
                    group_block = [True]*4 + candidate_button_options[candidate_option]
                else:
                    # For other energies: all False
                    group_block = [False] * (4 + 4 * num_candidates_effective)
                full_vec.extend(group_block)
            
            return full_vec

        candidate_buttons = []
        # "All" option: show all candidate traces in the current energy group.
        # FIXED: Use current energy instead of hardcoded default
        all_vis = []
        for group_idx in range(len(energy_keys)):
            if group_idx == energy_keys.index(energy_key):  # Current energy
                all_vis.extend([True]*4 + [True]*(4*num_candidates_effective))
            else:
                all_vis.extend([False] * (4 + 4 * num_candidates_effective))
        
        candidate_buttons.append(dict(
            label="All",
            method="update",
            args=[{"visible": all_vis}, {"annotations": []}]
        ))
        
        # Now add individual candidate (rank) buttons that also update the annotation.
        for i in range(num_candidates_effective):
            title = f"Candidate Rank {i+1}: SA Score: {sa_score[i]:.2f}, Entropy Score: {ent_score[i]:.2f}"
            if second_smi is not None:
                title += f"\n SMILES: {second_smi[i]}"
            annotation = dict(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=title,
                showarrow=False,
                font=dict(size=14)
            )
            # FIXED: Pass current energy key to respect user's energy selection
            candidate_buttons.append(dict(
                label=f"Rank {i+1}",
                method="update",
                args=[{"visible": full_visible_vector_for_candidate(i, energy_key)},
                      {"annotations": [annotation]}]
            ))
        updatemenus.append({
            "active": 1,
            "buttons": candidate_buttons,
            "x": 1.1,
            "y": 0.6
        })
        
    # --------------------------------------------------------
    # Build color update dropdown for matched peaks.
    # --------------------------------------------------------
    if num_candidates_effective > 0:
        if not candidate_dropdown:
            second_matched_indices = candidate_matched_indices
        else:
            second_matched_indices = []
            for indices in candidate_indices_by_candidate:
                second_matched_indices.extend(indices)
        default_colors = (["blue"] * len(primary_matched_indices)) + (["red"] * len(second_matched_indices))
        combined_matched_indices = tuple(primary_matched_indices + second_matched_indices)
        buttons_color = [
            dict(
                label="Default",
                method="restyle",
                args=[{"line.color": default_colors, "marker.color": default_colors}, combined_matched_indices]
            ),
            dict(
                label="Color Match Peaks",
                method="restyle",
                args=[{"line.color": "green", "marker.color": "green"}, combined_matched_indices]
            )
        ]
        updatemenus.append({
            "active": 0,
            "buttons": buttons_color,
            "x": 1.1,
            "y": 0.8
        })

    fig.update_layout(
        updatemenus=updatemenus,
        xaxis_title="m/z",
        yaxis_title="Intensity"
    )

    # --------------------------------------------------------
    # Set default annotation.
    # For multiple candidates, default is candidate rank 1 in the default energy group.
    # For a single candidate, use its score (index 0).
    # --------------------------------------------------------
    if (sa_score is not None) or (ent_score is not None):
        if candidate_dropdown:
            annotation_text = f"Candidate Rank 1: SA Score: {sa_score[0]:.2f}, Entropy Score: {ent_score[0]:.2f}"
        else:
            annotation_text = ""
            if sa_score is not None and ent_score is not None:
                annotation_text = f"SA Score: {sa_score[0]:.2f}, Entropy Score: {ent_score[0]:.2f}"
            elif sa_score is not None:
                annotation_text = f"SA Score: {sa_score[0]:.2f}"
            elif ent_score is not None:
                annotation_text = f"Entropy Score: {ent_score[0]:.2f}"
        if annotation_text:
            fig.update_layout(annotations=[dict(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=annotation_text,
                showarrow=False,
                font=dict(size=14)
            )])

    # # export to HTML to check - briefly
    # time = int(np.round(1000*np.random.rand()))
    # fig.write_html(f"to_delete/debug_{time}.html")
    # # see if it shows up properly in html, at least
    

    return fig


def plot_spectrum_new(
    spec_dict: Dict[str, np.ndarray],
    second_spec: Optional[Dict[str, np.ndarray]] = None,
    show_text: bool = False,
    show_dropdown: bool = False,
) -> go.Figure:
    """Create a simple multi-energy spectrum comparison plot.

    A simplified alternative to plot_spectrum that displays all collision
    energies as separate subplots in a single figure.

    Args:
        spec_dict: Primary spectrum as dict mapping collision energy (str) to
            2D numpy array of shape (n_peaks, 2) with [m/z, intensity] columns.
        second_spec: Optional secondary spectrum for comparison (same format).
        show_text: If True, display m/z labels on peaks.
        show_dropdown: If True, add dropdown to toggle primary/secondary/both.

    Returns:
        Plotly Figure with subplots for each collision energy.
    """
    from plotly.subplots import make_subplots
    energy_keys = list(spec_dict.keys())
    n = len(energy_keys)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=[f"Energy {k}" for k in energy_keys])
    for i, key in enumerate(energy_keys):
        arr1 = spec_dict[key]
        mz1 = arr1[:, 0]
        intensity1 = arr1[:, 1]
        text1 = [str(round(m, 2)) for m in mz1] if show_text else None
        fig.add_trace(
            go.Scatter(
                x=mz1, y=intensity1, mode='lines+markers'+('+text' if show_text else ''),
                line=dict(color='blue'), name=f'primary_{key}', text=text1, textposition='top center'
            ),
            row=i+1, col=1
        )
        if second_spec is not None and key in second_spec:
            arr2 = second_spec[key]
            mz2 = arr2[:, 0]
            intensity2 = arr2[:, 1]
            text2 = [str(round(m, 2)) for m in mz2] if show_text else None
            fig.add_trace(
                go.Scatter(
                    x=mz2, y=intensity2, mode='lines+markers'+('+text' if show_text else ''),
                    line=dict(color='red'), name=f'secondary_{key}', text=text2, textposition='top center'
                ),
                row=i+1, col=1
            )
    fig.update_layout(xaxis_title="m/z", yaxis_title="Intensity")
    # Optionally add dropdown to toggle all primary/secondary/both for all energies
    if show_dropdown and second_spec is not None:
        ntraces = 2 * n
        buttons = [
            dict(label="Primary", method="update", args=[{"visible": [i%2==0 for i in range(ntraces)]}]),
            dict(label="Secondary", method="update", args=[{"visible": [i%2==1 for i in range(ntraces)]}]),
            dict(label="Both", method="update", args=[{"visible": [True]*ntraces}]),
        ]
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    x=1.1,
                    y=1.15,
                    active=2,
                    buttons=buttons,
                )
            ]
        )
    return fig




import numpy as np
import wandb

if __name__ == "__main__":

    # Minimal test data: one energy, one candidate, 3 peaks each
    spec_dict = {
        '10': np.array([
            [100, 0.5],
            [150, 1.0],
            [200, 0.2],
        ])
    }

    # --- Plotting test functions of increasing complexity ---
    def test_plot_basic():
        # Single spectrum, no secondary, no text, no dropdown
        spec = {'10': np.array([[100, 0.5], [150, 1.0], [200, 0.2]])}
        fig = plot_spectrum_new(spec)
        return fig

    def test_plot_basic_text():
        # Single spectrum, with text labels
        spec = {'10': np.array([[100, 0.5], [150, 1.0], [200, 0.2]])}
        fig = plot_spectrum_new(spec, show_text=True)
        return fig

    def test_plot_primary_secondary():
        # Single spectrum, with secondary, no text, no dropdown
        spec = {'10': np.array([[100, 0.5], [150, 1.0], [200, 0.2]])}
        sec = {'10': np.array([[100, 0.4], [150, 0.9], [210, 0.1]])}
        fig = plot_spectrum_new(spec, second_spec=sec)
        return fig

    def test_plot_primary_secondary_text():
        # Single spectrum, with secondary, with text
        spec = {'10': np.array([[100, 0.5], [150, 1.0], [200, 0.2]])}
        sec = {'10': np.array([[100, 0.4], [150, 0.9], [210, 0.1]])}
        fig = plot_spectrum_new(spec, second_spec=sec, show_text=True)
        return fig

    def test_plot_primary_secondary_dropdown():
        # Single spectrum, with secondary, with dropdown
        spec = {'10': np.array([[100, 0.5], [150, 1.0], [200, 0.2]])}
        sec = {'10': np.array([[100, 0.4], [150, 0.9], [210, 0.1]])}
        fig = plot_spectrum_new(spec, second_spec=sec, show_dropdown=True)
        return fig

    def test_plot_multi_energy():
        # Multiple energies, primary only
        spec = {
            '10': np.array([[100, 0.5], [150, 1.0], [200, 0.2]]),
            '20': np.array([[110, 0.3], [160, 0.8], [210, 0.5]])
        }
        fig = plot_spectrum_new(spec)
        return fig

    def test_plot_multi_energy_secondary():
        # Multiple energies, with secondary, with dropdown
        spec = {
            '10': np.array([[100, 0.5], [150, 1.0], [200, 0.2]]),
            '20': np.array([[110, 0.3], [160, 0.8], [210, 0.5]])
        }
        sec = {
            '10': np.array([[100, 0.4], [150, 0.9], [210, 0.1]]),
            '20': np.array([[110, 0.2], [160, 0.7], [220, 0.3]])
        }
        fig = plot_spectrum_new(spec, second_spec=sec, show_dropdown=True)
        return fig

    def test_plot_original():
        # Original complex plot for reference
        spec = {'10': np.array([[100, 0.5], [150, 1.0], [200, 0.2]])}
        sec = {'10': np.array([[100, 0.4], [150, 0.9], [210, 0.1]])}
        fig = plot_spectrum(spec, second_spec=sec)
        return fig

    # --- Run and log all tests ---
    wandb.init(project="test-plotly")
    plot_tests = {
        "basic": test_plot_basic(),
        "basic_text": test_plot_basic_text(),
        "primary_secondary": test_plot_primary_secondary(),
        "primary_secondary_text": test_plot_primary_secondary_text(),
        "primary_secondary_dropdown": test_plot_primary_secondary_dropdown(),
        "multi_energy": test_plot_multi_energy(),
        "multi_energy_secondary": test_plot_multi_energy_secondary(),
        "original_complex": test_plot_original(),
    }
    wandb.log(plot_tests)
