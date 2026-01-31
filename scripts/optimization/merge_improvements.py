
import pandas as pd
import glob
import os
from decimal import Decimal, getcontext
from shapely.strtree import STRtree
from shapely.ops import unary_union
from santa2025.metric import ChristmasTree, ParticipantVisibleError, scale_factor

# Set precision matches metric.py
getcontext().prec = 25

def get_group_scores(df):
    """
    Returns a dict: group_id -> {'score': float, 'valid': bool, 'rows': DataFrame}
    """
    # Preprocess like metric.py
    data_cols = ["x", "y", "deg"]
    df_clean = df.copy()
    
    # Ensure strings and handle 's' prefix
    for c in data_cols:
        df_clean[c] = df_clean[c].astype(str)
        # Check if s prefix exists on all (strict) or any
        # The metric.py raises error if ANY don't have s.
        # We assume input might or might not.
        # But bbox3 outputs usually have it.
        # robust strip
        df_clean[c] = df_clean[c].apply(lambda x: x[1:] if x.startswith('s') else x)

    df_clean["tree_count_group"] = df_clean["id"].astype(str).str.split("_").str[0]
    
    results = {}
    
    for group, df_group in df_clean.groupby("tree_count_group"):
        # Keep original rows for saving (with 's' prefix if present in original df)
        # Actually simplest is to reconstruct 's' prefix at save time if we parse it out.
        # But to avoid precision loss, we should cache the ORIGINAL rows corresponding to this group.
        # We can look them up in original df using index.
        original_rows = df.loc[df_group.index].copy()
        
        # Validation Logic from metric.py
        try:
            num_trees = len(df_group)
            placed_trees = []
            for _, row in df_group.iterrows():
                placed_trees.append(ChristmasTree(row["x"], row["y"], row["deg"]))
            
            all_polygons = [p.polygon for p in placed_trees]
            r_tree = STRtree(all_polygons)
            
            valid = True
            for i, poly in enumerate(all_polygons):
                indices = r_tree.query(poly)
                for index in indices:
                    if index == i:
                        continue
                    if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                results[group] = {'score': float('inf'), 'valid': False, 'rows': original_rows}
                continue

            # Score Logic
            bounds = unary_union(all_polygons).bounds
            side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
            group_score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees)
            
            results[group] = {'score': float(group_score), 'valid': True, 'rows': original_rows}
            
        except Exception as e:
            # Catch bad coords etc
            print(f"Error checking group {group}: {e}")
            results[group] = {'score': float('inf'), 'valid': False, 'rows': original_rows}
            
    return results

def main():
    baseline_path = "results/submissions/best_submission.csv"
    candidate_patterns = [
        "results/kaggle_cycle2_v8/combined_opt/bbox3/*.csv",
        "results/public_nctuan/submission.csv",
        "results/public_hvanphucs112/submission.csv",
        "results/cycle3/combined/combined_opt/bbox3/*.csv",
        "results/cycle3/combined/combined_opt/gb_sa/best_submission.csv",
        "results/cycle3/combined/combined_opt/best_submission.csv",
        "results/cycle3/small/attack_small/best_submission.csv",
        "results/cycle3/large/attack_large/best_submission.csv"
    ]
    output_path = "results/submissions/cycle3_final_merge.csv"
    
    print(f"Loading baseline: {baseline_path}")
    baseline_df = pd.read_csv(baseline_path)
    current_best = get_group_scores(baseline_df)
    
    # Validate baseline
    invalid_baseline = [g for g, res in current_best.items() if not res['valid']]
    if invalid_baseline:
        print(f"WARNING: Baseline has invalid groups: {invalid_baseline}")
    
    baseline_total = sum(res['score'] for res in current_best.values() if res['valid'])
    print(f"Baseline Total Score (Valid Only): {baseline_total}")

    candidates = []
    for pat in candidate_patterns:
        candidates.extend(glob.glob(pat))
    print(f"Found {len(candidates)} candidate files.")
    
    for cand_path in candidates:
        print(f"Processing {cand_path}...")
        cand_df = pd.read_csv(cand_path)
        cand_res = get_group_scores(cand_df)
        
        improved_groups = 0
        for group, res in cand_res.items():
            if not res['valid']:
                continue
            
            # Check if group exists in baseline (it should)
            if group not in current_best:
                print(f"New group {group} found in candidate?")
                current_best[group] = res
                continue
                
            current_score = current_best[group]['score']
            new_score = res['score']
            
            # Update if better
            if new_score < current_score - 1e-12: # epsilon improvement
                diff = current_score - new_score
                # print(f"  Group {group} improved: {current_score} -> {new_score} (-{diff})")
                current_best[group] = res
                improved_groups += 1
                
        if improved_groups > 0:
            print(f"  -> Found {improved_groups} improved groups in this file.")
            
    # Assemble final submission
    print("Assembling final merged submission...")
    final_rows = []
    
    # Sort by group ID integer
    sorted_groups = sorted(current_best.keys(), key=lambda x: int(x))
    
    final_score = 0
    for group in sorted_groups:
        res = current_best[group]
        final_rows.append(res['rows'])
        if res['valid']:
            final_score += res['score']
    
    final_df = pd.concat(final_rows)
    # Ensure correct column order
    final_df = final_df[["id", "x", "y", "deg"]]
    
    print(f"Final Merged Score: {final_score}")
    print(f"Improvement over baseline: {baseline_total - final_score}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
