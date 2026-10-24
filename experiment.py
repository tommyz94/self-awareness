import argparse
import os
import json
from collections import Counter
import pandas as pd
import numpy as np
import re


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute aggregate metrics from exported run directories."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="outputs",
        help="Root directory to scan. Defaults to ./outputs",
    )
    parser.add_argument(
        "--experiments",
        default="e1,e2,e3,e5,e7",
        help="Comma-separated experiment ids to process. Default: e1,e2,e3,e5,e7.",
    )
    parser.add_argument(
        "--csv-out",
        default="result.csv",
        help="Output CSV path. Default: result.csv",
    )
    return parser.parse_args()


def compute_metrics(base_dir="outputs", experiments_to_process=None, csv_out="result.csv"):
    if experiments_to_process is None:
        experiments_to_process = ["e1", "e2", "e3", "e5", "e7"]
    
    # Textual metric patterns
    srr_words = [r'\bi\b', r'\bme\b', r'\bmy\b', r'\bmine\b', r'\bmyself\b', r"\bi'm\b", r"\bi've\b", r"\bi'll\b"]
    srr_pattern = re.compile('|'.join(srr_words), re.IGNORECASE)
    
    eul_words = [r'\bthink\b', r'\bappear[s]?\b', r'\bmight\b', r'\bmaybe\b', r'\bprobably\b', 
                 r'\bguess\b', r'\bperhaps\b', r'\bseem[s]?\b', r'\bunsure\b', r'\bnot sure\b']
    eul_pattern = re.compile('|'.join(eul_words), re.IGNORECASE)

    records = []

    def _normalize_label(value, fallback):
        if value in [None, "", "null", "unknown"]:
            return fallback
        return str(value)

    def _build_confusion_json(group, prediction_col, missing_label):
        labels = set(group["agent_color"].dropna().astype(str).tolist())
        labels.update(
            _normalize_label(value, missing_label)
            for value in group[prediction_col].tolist()
        )
        labels.add(missing_label)
        ordered_labels = sorted(labels)
        matrix = {
            agent: {label: 0 for label in ordered_labels}
            for agent in ordered_labels
        }
        for _, record in group.iterrows():
            agent = _normalize_label(record["agent_color"], "unknown_agent")
            pred = _normalize_label(record[prediction_col], missing_label)
            if agent not in matrix:
                matrix[agent] = {label: 0 for label in ordered_labels}
            if pred not in matrix[agent]:
                for row in matrix.values():
                    row[pred] = 0
                ordered_labels.append(pred)
            matrix[agent][pred] += 1

        row_norm = {}
        for agent, counts in matrix.items():
            total = sum(counts.values())
            if total == 0:
                row_norm[agent] = {label: 0.0 for label in counts}
            else:
                row_norm[agent] = {
                    label: counts[label] / total
                    for label in counts
                }
        return (
            json.dumps(matrix, sort_keys=True),
            json.dumps(row_norm, sort_keys=True),
            json.dumps(ordered_labels),
        )

    # Walk everything inside outputs/
    for root, dirs, files in os.walk(base_dir):
        # Identify if current 'root' is an experiment directory
        exp = os.path.basename(root)
        if exp not in experiments_to_process:
            continue
            
        # Inside this experiment directory, walk to find run*.json files
        for sub_root, sub_dirs, sub_files in os.walk(root):
            for file in sub_files:
                if file.startswith("run") and file.endswith(".json"):
                    file_path = os.path.join(sub_root, file)
                    
                    # Extract provider and model relative to the experiment dir (root)
                    rel_path = os.path.relpath(sub_root, root)
                    parts = rel_path.split(os.sep)
                    # Expected: <provider>/<model>/<level>/<datetime>/run_<number>
                    if len(parts) >= 2:
                        provider = parts[0]
                        model_name = parts[1]
                        
                        # Apply model name mappings
                        model_mapping = {
                            "google_gemma-4-26b-a4b-it_nitro": "gemma4_26b",
                            "google_gemma-4-31b-it_nitro": "gemma4_31b",
                            "mistralai_ministral-14b-2512_nitro": "ministral-3_14b"
                        }
                        model_name = model_mapping.get(model_name, model_name)
                        if model_name == "moonshotai_kimi-k2.6_nitro":
                            continue
                    else:
                        continue
                        
                    with open(file_path, 'r') as f:
                        try:
                            data = json.load(f)
                        except:
                            continue
                            
                    traj = data.get("step_records", [])
                        
                    agent_color = data.get("agent_color")
                    seed = data.get("run_setup", {}).get("seed")
                    
                    # Group 1: Task Success
                    # TSA = color accuracy against the ground-truth target for cube tasks.
                    # For e7, TSA remains final self-identification accuracy because there is
                    # no cube-selection task output to score against.
                    tsa = 0
                    done_step = None
                    final_ident_color = None
                    if traj:
                        final_ident_color = traj[-1].get("identification", {}).get("my_color")
                    final_selected_cube = None
                    for step in traj:
                        if step.get("action") == "done":
                            done_step = step
                            ident_color = step.get("identification", {}).get("my_color")
                            final_ident_color = ident_color
                            if exp in ["e1", "e2", "e3", "e5", "e6"]:
                                selected_cube = step.get("selected_cube")
                                final_selected_cube = selected_cube
                                if selected_cube is not None and agent_color is not None and selected_cube == agent_color:
                                    tsa = 1
                            elif exp == "e7":
                                if ident_color is not None and agent_color is not None and ident_color == agent_color:
                                    tsa = 1
                            break
                            
                    mcr = 0
                    for step in traj:
                        if step.get("action") == "done":
                            break
                        if step.get("mirror_visible", False) or step.get("ego_reflection_visible", False):
                            mcr = 1
                            break
                            
                    ttd = len(traj)
                    if done_step is not None:
                        for i, step in enumerate(traj):
                            if step.get("action") == "done":
                                ttd = i + 1
                                break
                    
                    # Group 2: Mirror Interaction Quality
                    mgd = sum(1 for step in traj if step.get("mirror_visible", False) or step.get("ego_reflection_visible", False))
                    
                    mtato = None
                    if tsa == 1:
                        mtato = 0
                        first_mirror_step = None
                        for i, step in enumerate(traj):
                            if step.get("mirror_visible", False) or step.get("ego_reflection_visible", False):
                                first_mirror_step = i
                                break
                        
                        done_idx = None
                        for i, step in enumerate(traj):
                            if step.get("action") == "done":
                                done_idx = i
                                break
                                
                        if first_mirror_step is not None and done_idx is not None and first_mirror_step < done_idx:
                            mtato = 1
                        
                    # Group 3: Text Based
                    srr_count = 0
                    eul_count = 0
                    for step in traj:
                        text = (step.get("reasoning") or "") + " " + (step.get("summary") or "")
                        if srr_pattern.search(text):
                            srr_count += 1
                        if eul_pattern.search(text):
                            eul_count += 1
                            
                    srr = srr_count / len(traj) if traj else 0
                    eul = eul_count / len(traj) if traj else 0
                    
                    caal = 0
                    for step in traj:
                        if step.get("action") == "done":
                            break
                        if step.get("identification", {}).get("my_color") == agent_color:
                            caal = 1
                            break
                            
                    rcs = 0 if not traj else 1
                    identified_color = None
                    for step in traj:
                        color = step.get("identification", {}).get("my_color")
                        if color and color not in ["unknown", "null", None]:
                            if identified_color is None:
                                identified_color = color
                            elif identified_color != color:
                                rcs = 0
                                break
                                
                    confabulation = 0
                    first_mirror_idx = -1
                    for i, step in enumerate(traj):
                        if step.get("mirror_visible", False) or step.get("ego_reflection_visible", False):
                            first_mirror_idx = i
                            break
                    
                    for i, step in enumerate(traj):
                        color = step.get("identification", {}).get("my_color")
                        if color and color not in ["unknown", "null", None]:
                            if first_mirror_idx == -1 or i < first_mirror_idx:
                                confabulation = 1
                            break

                    self_correction = np.nan
                    self_correction_applicable = 0
                    if first_mirror_idx != -1 and agent_color not in [None, "", "null", "unknown"]:
                        first_pre_mirror_guess = None
                        for i, step in enumerate(traj):
                            if i >= first_mirror_idx:
                                break
                            color = step.get("identification", {}).get("my_color")
                            if color and color not in ["unknown", "null", None]:
                                first_pre_mirror_guess = color
                                break

                        if (
                            first_pre_mirror_guess is not None
                            and first_pre_mirror_guess != agent_color
                        ):
                            self_correction_applicable = 1
                            self_correction = 0.0
                            for i, step in enumerate(traj):
                                if i < first_mirror_idx:
                                    continue
                                if step.get("identification", {}).get("my_color") == agent_color:
                                    self_correction = 1.0
                                    break
                                if step.get("action") == "done":
                                    break
                    
                    records.append({
                        "experiment_id": exp,
                        "model_name": model_name,
                        "run_completed": 1 if done_step is not None else 0,
                        "tsa": tsa,
                        "mcr": mcr,
                        "ttd": ttd,
                        "mgd": mgd,
                        "mtato": mtato,
                        "srr": srr,
                        "caal": caal,
                        "eul": eul,
                        "rcs": rcs,
                        "confabulation": confabulation,
                        "self_correction": self_correction,
                        "self_correction_applicable": self_correction_applicable,
                        "seed": seed,
                        "agent_color": agent_color,
                        "final_selected_cube": final_selected_cube,
                        "final_ident_color": final_ident_color,
                    })
                    
    df = pd.DataFrame(records)
    if df.empty:
        print("No valid runs found in outputs.")
        return
        
    # Calculate aggregations
    metrics_to_agg = [
        "tsa",
        "mcr",
        "ttd",
        "mgd",
        "mtato",
        "srr",
        "caal",
        "eul",
        "rcs",
        "confabulation",
        "self_correction",
    ]
    
    aggregated = []
    for (exp, model), group in df.groupby(['experiment_id', 'model_name']):
        total_runs = len(group)
        unique_seeds = group['seed'].nunique()
        row = {
            'experiment_id': exp, 
            'model_name': model,
            'total_runs': total_runs,
            'unique_seeds': unique_seeds,
            'avg_runs_per_seed': total_runs / unique_seeds if unique_seeds > 0 else 0
        }
        
        # Completion-conditioned success metrics
        completed_runs = group[group['run_completed'] == 1]
        row['TSA-COMPLETE_mean'] = completed_runs['tsa'].mean() if len(completed_runs) > 0 else np.nan
        row['TSA-COMPLETE_std'] = completed_runs['tsa'].std() if len(completed_runs) > 1 else 0.0
        row['TSA-COMPLETE_se'] = completed_runs['tsa'].sem() if len(completed_runs) > 1 else 0.0

        action_counts_json, action_row_norm_json, action_labels_json = _build_confusion_json(
            group,
            prediction_col="final_selected_cube",
            missing_label="no_selection",
        )
        ident_counts_json, ident_row_norm_json, ident_labels_json = _build_confusion_json(
            group,
            prediction_col="final_ident_color",
            missing_label="unknown_identification",
        )
        row["action_confusion_labels_json"] = action_labels_json
        row["ident_confusion_labels_json"] = ident_labels_json
        row["action_confusion_counts_json"] = action_counts_json
        row["action_confusion_row_norm_json"] = action_row_norm_json
        row["ident_confusion_counts_json"] = ident_counts_json
        row["ident_confusion_row_norm_json"] = ident_row_norm_json
        row["SELF_CORRECTION_applicable_runs"] = int(group["self_correction_applicable"].sum())
        row["SELF_CORRECTION_applicable_rate"] = (
            group["self_correction_applicable"].mean() if len(group) > 0 else np.nan
        )
        
        for m in metrics_to_agg:
            m_data = group[m].dropna()
            row[f'{m.upper()}_mean'] = m_data.mean() if len(m_data) > 0 else np.nan
            row[f'{m.upper()}_std'] = m_data.std() if len(m_data) > 1 else 0.0
            row[f'{m.upper()}_se'] = m_data.sem() if len(m_data) > 1 else 0.0
            
        aggregated.append(row)
        
    result_df = pd.DataFrame(aggregated)
    # Sort for consistent output
    result_df = result_df.sort_values(by=['experiment_id', 'model_name'])
    result_df.to_csv(csv_out, index=False)
    print(
        f"Metrics computation complete. Evaluated {len(df)} runs across "
        f"{len(result_df)} model-experiment combinations. Results saved to {csv_out}"
    )

if __name__ == "__main__":
    args = _parse_args()
    experiments = [part.strip() for part in args.experiments.split(",") if part.strip()]
    compute_metrics(
        base_dir=args.base_dir,
        experiments_to_process=experiments,
        csv_out=args.csv_out,
    )
