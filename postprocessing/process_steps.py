
import argparse
import glob
import os
import re
import ast
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem

def parse_log_file(log_path):
    """
    Parses a FOAM output.log file.
    """
    with open(log_path, 'r') as f:
        lines = f.readlines()

    config = {}
    steps = []
    
    current_step_yaml = []
    in_step_block = False

    # Regex to detect log line start
    log_start_re = re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+[A-Z]+:\s+(.*)$')

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        match = log_start_re.match(line)
        if match:
            # New log entry
            content = match.group(1)
            
            # Process previous block if any
            if in_step_block:
                if current_step_yaml:
                    # Fix: The first line often has a leading space which breaks YAML if subsequent lines don't.
                    current_step_yaml[0] = current_step_yaml[0].lstrip()
                    try:
                        step_data = yaml.safe_load("\n".join(current_step_yaml))
                        if isinstance(step_data, dict):
                             steps.append(step_data)
                    except yaml.YAMLError as e:
                        # Debugging: print first few lines of failed block
                        print(f"Warning: Failed to parse step YAML block (lines {len(current_step_yaml)}). Error: {e}")
                        print("Block snippet:", current_step_yaml[:5])
                
                current_step_yaml = []
                in_step_block = False

            # Check for batch statistics start
            if "Batch statistics" in content:
                in_step_block = True
                continue
            
            # Check for target smiles
            if "Canonicalized smiles:" in content:
                 parts = content.split("smiles: ", 1)
                 if len(parts) == 2:
                     config["target_smiles"] = parts[1].strip()

            # Check for single-line config (e.g. "batch_size: 32")
            if not steps and ": " in content and not in_step_block:
                 parts = content.split(": ", 1)
                 if len(parts) == 2:
                     config[parts[0]] = parts[1]

        else:
            # Not a timestamped line
            if in_step_block:
                current_step_yaml.append(line.rstrip())
            elif not steps:
                # Likely part of the initial config dump (no timestamp)
                # valid config lines look like "key: value"
                if ": " in line:
                    parts = line.split(": ", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip()
                        try:
                            val = yaml.safe_load(v)
                        except:
                            val = v
                        config[k] = val

    # Handle last block
    if in_step_block and current_step_yaml:
        # Fix: The first line often has a leading space
        current_step_yaml[0] = current_step_yaml[0].lstrip()
        try:
             step_data = yaml.safe_load("\n".join(current_step_yaml))
             if isinstance(step_data, dict):
                 steps.append(step_data)
        except yaml.YAMLError as e:
             print(f"Warning: Failed to parse last step YAML block. Error: {e}")
             print("Block snippet:", current_step_yaml[:5])

    return config, steps

def get_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToInchiKey(mol)
    except:
        pass
    return None

def compute_derived_metrics(config, steps):
    """
    Computes derived metrics for each step.
    Adds columns to the steps dataframe (or list of dicts).
    """
    target_smiles = config.get("target_smiles")
    target_inchikey = None
    if target_smiles:
        target_inchikey = get_inchikey(target_smiles)

    for step in steps:
        # 1. best_entropy
        # extracted from TopNDSScore@1 (list of scores). Assuming index 0 is entropy/cosine.
        if "TopNDSScore" in step and "TopNDSScore@1" in step["TopNDSScore"]:
             scores = step["TopNDSScore"]["TopNDSScore@1"]
             if isinstance(scores, list) and len(scores) > 0:
                 step["best_entropy"] = scores[0]
                 step['SA_of_best_entropy'] = scores[1]
             else:
                 step["best_entropy"] = 0.0
                 step['SA_of_best_entropy'] = 0.0
        else:
             step["best_entropy"] = 0.0
             step['SA_of_best_entropy'] = 0.0
       
        # 2. top1_match, top10_match
        # Check NDSBestMol keys
        top1_match = False
        top10_match = False
        
        if target_inchikey and "NDSBestMol" in step:
            # Check up to top 10
            for k in range(1, 11):
                key = f"NDSBestMol@{k}"
                if key in step["NDSBestMol"]:
                    smi = step["NDSBestMol"][key]
                    # Remove trailing ' *' if present (some loggers add it for match)
                    if isinstance(smi, str):
                        smi = smi.split(" *")[0]
                        ikey = get_inchikey(smi)
                        if ikey == target_inchikey:
                            top10_match = True
                            if k == 1:
                                top1_match = True
            
        step["top1_match"] = 1 if top1_match else 0
        step["top10_match"] = 1 if top10_match else 0
        
        # 3. Flattener for nested keys to make DataFrame creation easier
        if "Meta" in step:
            for k, v in step["Meta"].items():
                step[f"Meta.{k}"] = v
        
        if "NDSBestMol" in step:
             for k, v in step["NDSBestMol"].items():
                 # Flatten Tanimoto and other interesting scalar metrics
                 if "tani" in k or "entropy" in k or "SA" in k:
                     step[f"NDSBestMol.{k}"] = v
        
        if "InchiKeyMatch" in step:
            # Flatten InchiKeyMatch keys if needed
             for k, v in step["InchiKeyMatch"].items():
                 step[f"InchiKeyMatch.{k}"] = v

    return steps

def build_dataframes(results_dir):
    """
    Iterates over all subdirectories in results_dir.
    Returns:
        steps_df (pd.DataFrame): DataFrame containing all steps from all runs.
        summaries_df (pd.DataFrame): DataFrame containing summary metrics (final step usually).
    """
    all_steps = []
    summaries = []
    
    # walk through results_dir
    log_files = glob.glob(os.path.join(results_dir, "**", "output.log"), recursive=True)
    
    print(f"Found {len(log_files)} log files in {results_dir}")
    
    for log_path in tqdm(log_files, desc="Parsing logs"):
        # parse metadata:
        metadata_path = os.path.join(os.path.dirname(log_path), "oracle_metadata.yaml")
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)

        run_name = os.path.basename(os.path.dirname(log_path))
        try:
            config, steps = parse_log_file(log_path)
            if not steps:
                continue
                
            steps = compute_derived_metrics(config, steps)
            
            # Add run_name to each step
            for idx, s in enumerate(steps):
                s["run_name"] = run_name
                s["generation"] = idx
            
            all_steps.extend(steps)
            
            # Create a summary row (final step)
            # We will enrich this later with metrics @ k steps
            last_step = steps[-1].copy()
            last_step["run_name"] = run_name
            # Merge config into summary
            for k, v in config.items():
                if k not in last_step: # prefer step data if conflict
                    last_step[k] = v

            # Also merge metadata into summary
            for k, v in metadata.items():
                if k not in last_step: # prefer step data if conflict
                    last_step[k] = v
            summaries.append(last_step)
            
        except Exception as e:
            print(f"Error parsing {log_path}: {e}")

    steps_df = pd.DataFrame(all_steps)
    summaries_df = pd.DataFrame(summaries)
    
    if not summaries_df.empty:
        summaries_df.set_index("run_name", inplace=True)
        
    return steps_df, summaries_df



RENAME_METRIC_COLS = {
    "best_entropy": "top1_entropy",
    "InchiKeyMatch.InchiKeyMatch": "inchikey_match",
    "NDSBestMol.top_1_tani_similarity": "top1_tani",
    "NDSBestMol.top_k_max_tani_similarity": "top10_max_tani",
    "top1_match": "top1_match",
    "top10_match": "top10_match",
    "NDSBestMol.better_entropy_decoy": "better_entropy_decoy",
}

METRICS_TO_TRACK = list(RENAME_METRIC_COLS.values())


def _get_latest_per_run_at_k(steps_df, k, step_col, metrics_to_track=None):
    """
    For each run, get the latest step where step_col <= k.
    Returns a DataFrame indexed by run_name with the tracked metric columns.
    Returns None if no data available.
    """
    if metrics_to_track is None:
        metrics_to_track = METRICS_TO_TRACK

    if step_col not in steps_df.columns:
        return None

    subset = steps_df[steps_df[step_col] <= k]
    if subset.empty:
        return None

    subset = subset.sort_values(step_col)
    latest_at_k = subset.groupby("run_name").last()

    cols = [c for c in metrics_to_track + [step_col] if c in latest_at_k.columns]
    return latest_at_k[cols]


def get_metrics_at_k(steps_df, summaries_df,
                    k_steps=range(10), step_col="generation"):
    """
    Computes metrics at specific k steps and adds them to summaries_df.
    e.g. best_entropy@1000
    """

    timestep_summary = pd.DataFrame()
    if steps_df.empty or summaries_df.empty:
        return summaries_df, timestep_summary

    by_k = []
    for k in k_steps:
        latest_at_k = _get_latest_per_run_at_k(steps_df, k, step_col)
        if latest_at_k is None:
            continue
        
        print(steps_df.columns, METRICS_TO_TRACK)
        for metric in METRICS_TO_TRACK:
            if metric in latest_at_k.columns:
                col_name = f"{metric}@{k}"
                summaries_df[col_name] = latest_at_k[metric]

        avgs = latest_at_k.mean().to_dict()
        avgs['generation'] = k
        by_k.append(avgs)
    timestep_summary = pd.DataFrame(by_k)


    return summaries_df, timestep_summary


def bootstrap_ci_multimetric(df, metrics_name_to_col, n_boot=1000, ci=95, seed=42):
    """
    Bootstrap confidence intervals for the mean of multiple metrics.
    Resamples rows (runs) of df with replacement.

    Args:
        df: DataFrame where each row is one run's data at a single step.
        metrics_name_to_col: Dict mapping friendly metric names to df column names.
            e.g. {"top1_tani": "NDSBestMol.top_1_tani_similarity", ...}
        n_boot: Number of bootstrap resamples.
        ci: Confidence interval percentage (e.g. 95 or 99.9).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "{metric}_lower" and "{metric}_upper" for each metric.
    """
    rng = np.random.RandomState(seed)
    alpha = (100 - ci) / 2
    n = len(df)
    indices = rng.randint(0, n, size=(n_boot, n))

    result = {}
    for name, col in metrics_name_to_col.items():
        if col not in df.columns:
            continue
        values = df[col].values.astype(float)
        boot_means = np.nanmean(values[indices], axis=1)
        result[f"{name}_lower"] = np.percentile(boot_means, alpha)
        result[f"{name}_upper"] = np.percentile(boot_means, 100 - alpha)

    return result


def add_additional_columns(summary_df, steps_df):
    summary_df = summary_df.copy()
    summary_df['best_seed_bin'] = pd.cut(summary_df['best_seed_sim'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    summary_df['oracle_accuracy_bin'] =  pd.cut(summary_df['NDSBestMol.target_self_entropy'], bins=np.arange(0, 1.1, 0.2))
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Parse FOAM results into CSVs")
    parser.add_argument("input_path", help="Path to output.log file OR directory containing results")
    parser.add_argument("--output-dir", help="Directory to save CSVs. Default: same as input_path")
    parser.add_argument("--step-type", choices=["calls", "generations"], default="generations", help="Which step metric to use for @k aggregation (default: generations)")
    args = parser.parse_args()

    # Determine step column based on step-type
    if args.step_type == "calls":
        step_col = "Meta.Calls Made"
    else:
        step_col = "generation"

    if os.path.isdir(args.input_path):
        print(f"Building dataframes from directory: {args.input_path}")
        steps_df, summaries_df = build_dataframes(args.input_path)
    elif os.path.isfile(args.input_path):
        print(f"Parsing single log file: {args.input_path}")
        config, steps = parse_log_file(args.input_path)
        if steps:
            steps = compute_derived_metrics(config, steps)
            for s in steps:
                # for single file, single run
                s["run_name"] = "single_run"
            steps_df = pd.DataFrame(steps)

            summaries_df = pd.DataFrame([steps[-1]]) 
            summaries_df["run_name"] = "single_run"
            summaries_df.set_index("run_name", inplace=True)
        else:
             steps_df = pd.DataFrame()
             summaries_df = pd.DataFrame()
    else:
        print("Invalid input path")
        return


    # rename columns in steps as needed.
    steps_df = steps_df.rename(columns=RENAME_METRIC_COLS)

    # Create output directory: input_path/compiled. 
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_path), "compiled")
    os.makedirs(args.output_dir, exist_ok=True)

    if not steps_df.empty:
        print(f"Steps DataFrame: {steps_df.shape}")
        print(f"Summaries DataFrame: {summaries_df.shape}")
        
        # Enrich summaries
        summaries_df, timestep_summary = get_metrics_at_k(steps_df, summaries_df, step_col=step_col)
        summaries_df = add_additional_columns(summaries_df, steps_df)
        # cols_to_check = [c for c in summaries_df.columns if "@" in c]
        # if cols_to_check:
        #     print("Metrics at k stats:")
        #     print(summaries_df[cols_to_check].describe().T[["mean", "max"]])
        
        # Save CSVs
        output_csv_steps = os.path.join(args.output_dir, "steps.tsv")
        output_csv_summary = os.path.join(args.output_dir, "summary.tsv")
        timestep_summary.to_csv(os.path.join(args.output_dir, "timestep_summary.tsv"), sep="\t", index=False)
        steps_df.to_csv(output_csv_steps, sep="\t")
        summaries_df.to_csv(output_csv_summary, sep="\t")
        print(f"Saved extracted data to {output_csv_steps} and {output_csv_summary}")
    else:
        print("No steps found.")

if __name__ == "__main__":
    main()
