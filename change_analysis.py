import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ==================== COMPARTMENT DEFINITIONS ====================
COMPARTMENT_RULES = {
    "Primary Tumor/GTV": ["GTV", "PTV", "CTV", "Primary", "Tumor", "Mass"],
    "Lymph Nodes": ["Node", "LN", "Lymph", "Supraclavicular", "Mediastinal", "Hilar", "Axillary", "Inguinal"],
    "Bone Marrow": ["Marrow", "BM", "Vertebra", "Spine", "Pelvis", "Femur"],
    "Spleen": ["Spleen"],
    "Liver": ["Liver"],
    "Kidneys": ["Kidney", "Renal"],
    "Lungs": ["Lung"],
    "Thymus": ["Thymus"],
    "Parotids/Salivary": ["Parotid", "Submandibular", "Salivary"],
    "Brain/CNS": ["Brain", "Brainstem", "Spinal", "Optic", "Cochlea"],
    "Heart/Great Vessels": ["Heart", "Aorta", "Ventricle", "Atrium"],
    "Bowel/GI": ["Bowel", "Rectum", "Colon", "Small bowel", "Esophagus", "Stomach", "Duodenum"],
    "Muscle/Soft Tissue": ["Muscle", "Soft tissue", "Fascia"],
    "Bone/Skeletal": ["Bone", "Rib", "Sternum", "Mandible", "Maxilla"],
}

IMMUNE_COMPARTMENTS = {
    "Primary Immune": ["Bone Marrow", "Thymus"],
    "Secondary Immune": ["Spleen", "Lymph Nodes"],
    "Immune-Associated": ["Liver", "Lungs", "Bowel/GI"]
}


# ==================== HELPER FUNCTIONS ====================
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing file: " + path)
    
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = [col.replace('Ã‚', '') for col in df.columns]
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if encoding == encodings[-1]: 
                raise e
            continue
    
    return pd.read_csv(path, encoding='utf-8', errors='replace')


def normalize_name(name):
    n = str(name)
    n = n.replace("_RelativeCumVolume", "")
    n = n.replace("_Volume_cm3", "")
    n = n.replace("_Volume_mL", "")
    n = n.replace("_", " ").strip()
    n = re.sub(r"\s+", " ", n)
    return n


def assign_compartment(structure_name):
    s = structure_name.lower()
    for comp, words in COMPARTMENT_RULES.items():
        for w in words:
            if w.lower() in s:
                return comp
    return "Other"


def assign_immune_category(compartment):
    for category, comps in IMMUNE_COMPARTMENTS.items():
        if compartment in comps:
            return category
    return "Non-Immune"


def _resolve_timepoint_paths(paths_like):
    if isinstance(paths_like, dict):
        return paths_like

    base = os.path.abspath(paths_like)
    candidates = [
        os.path.join(base, "SUV_Data", "SUVbw_AllStructures.csv"),
        os.path.join(base, "generated_data", "SUV_Data", "SUVbw_AllStructures.csv"),
    ]
    for suv in candidates:
        if os.path.exists(suv):
            return {"suv_stats": suv}

    return {"suv_stats": candidates[0]}


def load_timepoint_data(paths_like):
    paths = _resolve_timepoint_paths(paths_like)
    suv_df = safe_read_csv(paths["suv_stats"])

    if "Structure Name" in suv_df.columns:
        suv_df["Structure_norm"] = suv_df["Structure Name"].apply(normalize_name)
    else:
        second_col = suv_df.columns[1]
        suv_df["Structure_norm"] = suv_df[second_col].apply(normalize_name)

    return suv_df


def compute_structure_deltas(curr_suv, prev_suv):
    integral_cols_curr = [col for col in curr_suv.columns if 'integral' in col.lower() and 'activity' in col.lower()]
    integral_cols_prev = [col for col in prev_suv.columns if 'integral' in col.lower() and 'activity' in col.lower()]
    
    integral_col_curr = integral_cols_curr[0] if integral_cols_curr else "Integral Activity (Bq·mL)"
    integral_col_prev = integral_cols_prev[0] if integral_cols_prev else "Integral Activity (Bq·mL)"
    
    merged = pd.merge(
        curr_suv[["Structure_norm", "Volume (mL)", "SUVmean", "SUVmax", integral_col_curr]],
        prev_suv[["Structure_norm", "Volume (mL)", "SUVmean", "SUVmax", integral_col_prev]],
        on="Structure_norm", suffixes=("_curr", "_prev")
    )

    out = pd.DataFrame()
    out["Structure"] = merged["Structure_norm"]
    out["Compartment"] = out["Structure"].apply(assign_compartment)
    out["Immune_Category"] = out["Compartment"].apply(assign_immune_category)

    out["Delta_Volume_mL"] = merged["Volume (mL)_curr"] - merged["Volume (mL)_prev"]
    out["Delta_SUVmean"] = merged["SUVmean_curr"] - merged["SUVmean_prev"]
    out["Delta_SUVmax"] = merged["SUVmax_curr"] - merged["SUVmax_prev"]
    out["Delta_Integral_Bq_mL"] = merged[f"{integral_col_curr}_curr"] - merged[f"{integral_col_prev}_prev"]

    out["Curr_Integral"] = merged[f"{integral_col_curr}_curr"]
    out["Prev_Integral"] = merged[f"{integral_col_prev}_prev"]

    def pct(curr, prev):
        if pd.notna(curr) and pd.notna(prev) and prev != 0:
            return (curr - prev) / prev * 100.0
        return np.nan

    out["Pct_Delta_Volume"] = [
        pct(merged.at[i, "Volume (mL)_curr"], merged.at[i, "Volume (mL)_prev"]) for i in merged.index
    ]
    out["Pct_Delta_SUVmean"] = [
        pct(merged.at[i, "SUVmean_curr"], merged.at[i, "SUVmean_prev"]) for i in merged.index
    ]
    out["Pct_Delta_SUVmax"] = [
        pct(merged.at[i, "SUVmax_curr"], merged.at[i, "SUVmax_prev"]) for i in merged.index
    ]
    out["Pct_Delta_Integral"] = [
        pct(merged.at[i, f"{integral_col_curr}_curr"], merged.at[i, f"{integral_col_prev}_prev"]) for i in merged.index
    ]

    return out.sort_values(by="Delta_Integral_Bq_mL", ascending=False, na_position="last").reset_index(drop=True)


def _ensure_compare_dir():
    out_dir = os.path.join("Comparisons")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ==================== KEY VISUALIZATIONS ====================

def mean_dose_drop(df, output_dir, generated_data_dir='generated_data', generated_data2_dir='generated_data2'):
    """
    Scatter plot showing whether mean dose correlates with change (drop) in integral activity.

    This version loads mean dose from DVH_Structure_Statistics.csv files in both
    generated_data/DVH and generated_data2/DVH folders, then plots mean dose vs drop.
    
    Parameters:
    -----------
    df : DataFrame
        Structure-level deltas produced by compute_structure_deltas
    output_dir : str
        Directory to save the output plot
    generated_data_dir : str
        Path to first generated_data folder (default: 'generated_data')
    generated_data2_dir : str
        Path to second generated_data2 folder (default: 'generated_data2')
    """
    
    # Load the delta data if not provided
    if df is None or (hasattr(df, 'empty') and df.empty):
        csv_path = os.path.join(output_dir, 'per_structure_deltas.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error loading delta data: {e}")
                return
        else:
            print("No delta data found")
            return

    metric_y = 'Delta_Integral_Bq_mL'
    
    if metric_y not in df.columns:
        print(f"Column '{metric_y}' not found in dataframe")
        return

    # Load mean dose data from both DVH folders
    mean_doses = {}
    
    # Load from generated_data/DVH
    dvh_stats_path1 = os.path.join(generated_data_dir, 'DVH', 'DVH_Structure_Statistics.csv')
    if os.path.exists(dvh_stats_path1):
        try:
            dvh_df1 = pd.read_csv(dvh_stats_path1)
            if 'Structure' in dvh_df1.columns and 'Mean Dose (Gy)' in dvh_df1.columns:
                for _, row in dvh_df1.iterrows():
                    structure = row['Structure']
                    mean_dose = row['Mean Dose (Gy)']
                    mean_doses[structure] = mean_dose
                print(f"Loaded {len(dvh_df1)} structures from {dvh_stats_path1}")
        except Exception as e:
            print(f"Error loading DVH stats from {dvh_stats_path1}: {e}")
    
    # Load from generated_data2/DVH (this will be the "previous" scan data)
    dvh_stats_path2 = os.path.join(generated_data2_dir, 'DVH', 'DVH_Structure_Statistics.csv')
    if os.path.exists(dvh_stats_path2):
        try:
            dvh_df2 = pd.read_csv(dvh_stats_path2)
            if 'Structure' in dvh_df2.columns and 'Mean Dose (Gy)' in dvh_df2.columns:
                # Store as previous mean dose
                for _, row in dvh_df2.iterrows():
                    structure = row['Structure']
                    mean_dose_prev = row['Mean Dose (Gy)']
                    # If structure already has current dose, calculate average
                    if structure in mean_doses:
                        # Average of current and previous
                        mean_doses[structure] = (mean_doses[structure] + mean_dose_prev) / 2
                    else:
                        mean_doses[structure] = mean_dose_prev
                print(f"Loaded {len(dvh_df2)} structures from {dvh_stats_path2}")
        except Exception as e:
            print(f"Error loading DVH stats from {dvh_stats_path2}: {e}")
    
    if not mean_doses:
        print("No mean dose data found in DVH folders")
        return
    
    # Add mean dose to the dataframe
    df = df.copy()
    df['Mean_Dose_Gy'] = df['Structure'].map(mean_doses)
    
    # Drop rows without both mean dose and delta integral
    s = df.dropna(subset=['Mean_Dose_Gy', metric_y]).copy()

    max_dose_threshold = 2.5

    if max_dose_threshold is not None:
        n_before = len(s)
        s = s[s['Mean_Dose_Gy'] <= max_dose_threshold].copy()
        n_after = len(s)
        print(f"Filtered to doses ≤ {max_dose_threshold} Gy: {n_before} → {n_after} structures")
    
    if len(s) == 0:
        print("No valid data points with both mean dose and delta integral")
        return
    
    print(f"Found {len(s)} structures with both mean dose and integral activity drop data")
    
    # Prepare data for plotting
    x = s['Mean_Dose_Gy'].astype(float).values
    # If Delta_Integral_Bq_mL is negative for a drop, invert sign to make drops positive
    y = -s[metric_y].astype(float).values

    # Compute Pearson correlation and linear fit
    try:
        from scipy import stats
        pearson_r, pearson_p = stats.pearsonr(x, y)
        slope, intercept, r_value, p_value_lin, std_err = stats.linregress(x, y)
    except Exception:
        # Lightweight fallback using numpy
        pearson_r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
        pearson_p = np.nan
        # linear fit
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception:
            slope, intercept = 0.0, np.mean(y) if len(y) > 0 else 0.0
        r_value = pearson_r
        p_value_lin = np.nan
        std_err = np.nan
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color by immune category if present
    colors = None
    if 'Immune_Category' in s.columns:
        cmap = {
            'Primary Immune': '#e74c3c', 
            'Secondary Immune': '#e67e22', 
            'Immune-Associated': '#f39c12', 
            'Non-Immune': '#3498db'
        }
        colors = [cmap.get(v, '#95a5a6') for v in s['Immune_Category']]
        
        # Create scatter plot with legend
        for category, color in cmap.items():
            mask = s['Immune_Category'] == category
            if mask.any():
                ax.scatter(x[mask], y[mask], s=70, c=color, edgecolor='black', 
                          alpha=0.9, label=category)
        # Add any uncategorized points
        mask = ~s['Immune_Category'].isin(cmap.keys())
        if mask.any():
            ax.scatter(x[mask], y[mask], s=70, c='#95a5a6', edgecolor='black', 
                      alpha=0.9, label='Other')
    else:
        ax.scatter(x, y, s=70, c='#2c7fb8', edgecolor='black', alpha=0.85)
    
    # Add regression line
    if not np.isnan(slope):
        xs = np.linspace(np.min(x), np.max(x), 200)
        ys = slope * xs + intercept
        ax.plot(xs, ys, color='#f03b20', lw=2.5, linestyle='--', 
               label=f'Linear fit (slope={slope:.3f})')
    
    # Labels and title
    ax.set_xlabel('Mean Dose (Gy)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drop in Integral Activity (positive = larger drop) (Bq·mL)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Does Higher Dose Correlate with Larger Drop in Integral Activity?', 
                 fontsize=14, fontweight='bold')
    
    # Annotate correlation info
    ann_x = 0.05
    ann_y = 0.95
    n_pts = len(x)
    
    if not np.isnan(pearson_r):
        ann_text = f'n={n_pts}  |  Pearson r = {pearson_r:.3f}'
        if not np.isnan(pearson_p):
            ann_text += f'\np = {pearson_p:.4f}'
        else:
            ann_text += '\np = n/a'
    else:
        ann_text = f'n={n_pts}'
    
    ax.text(ann_x, ann_y, ann_text,
            transform=ax.transAxes, fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'mean_dose_vs_drop.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Mean Dose range: {np.min(x):.2f} - {np.max(x):.2f} Gy")
    print(f"  Drop range: {np.min(y):.2f} - {np.max(y):.2f} Bq·mL")
    if not np.isnan(pearson_r):
        print(f"  Correlation: r = {pearson_r:.3f}, p = {pearson_p if not np.isnan(pearson_p) else 'n/a'}")
    
    return

def plot_waterfall_by_compartment(df, out_dir):
    """
    TRUE waterfall plot showing cumulative change in integral activity by compartment.
    Each bar shows how the total changes as we add each compartment.
    """
    metric = "Delta_Integral_Bq_mL"
    s = df.dropna(subset=[metric]).copy()
    if len(s) == 0:
        return

    # Calculate compartment totals and sort by magnitude
    comp_totals = s.groupby("Compartment")[metric].sum().sort_values(ascending=False)
    compartments = list(comp_totals.index)
    values = comp_totals.values
    
    # Identify immune compartments
    immune_comps = [c for cat in IMMUNE_COMPARTMENTS.values() for c in cat]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Calculate cumulative positions for waterfall
    cumulative = 0
    bar_data = []
    
    # Starting point
    bar_data.append({
        'x': 0,
        'bottom': 0,
        'height': 0,
        'label': 'Start',
        'color': 'lightgray',
        'value': 0
    })
    
    # Each compartment
    for i, (comp, val) in enumerate(zip(compartments, values)):
        is_immune = comp in immune_comps
        
        if val >= 0:
            # Positive change - bar goes up
            bar_data.append({
                'x': i + 1,
                'bottom': cumulative,
                'height': val,
                'label': comp,
                'color': '#e74c3c' if is_immune else '#3498db',
                'value': val
            })
        else:
            # Negative change - bar goes down
            bar_data.append({
                'x': i + 1,
                'bottom': cumulative + val,
                'height': abs(val),
                'label': comp,
                'color': '#c0392b' if is_immune else '#2980b9',
                'value': val
            })
        
        cumulative += val
    
    # Ending point
    bar_data.append({
        'x': len(compartments) + 1,
        'bottom': 0,
        'height': cumulative,
        'label': 'End Total',
        'color': '#2ecc71' if cumulative > 0 else '#e74c3c',
        'value': cumulative
    })
    
    # Draw bars
    for bd in bar_data:
        bar = ax.bar(bd['x'], bd['height'], bottom=bd['bottom'], 
                     color=bd['color'], alpha=0.8, edgecolor='black', linewidth=2, width=0.8)
        
        # Add value labels
        label_y = bd['bottom'] + bd['height']/2
        ax.text(bd['x'], label_y, f"{bd['value']:+.1f}", 
                ha='center', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw connector lines
    for i in range(len(bar_data) - 1):
        x1, x2 = bar_data[i]['x'], bar_data[i+1]['x']
        y = bar_data[i]['bottom'] + (bar_data[i]['height'] if bar_data[i]['value'] >= 0 else 0)
        ax.plot([x1 + 0.4, x2 - 0.4], [y, y], 'k--', linewidth=1.5, alpha=0.6)
    
    # Formatting
    ax.axhline(0, color='black', linewidth=2)
    ax.set_xticks([bd['x'] for bd in bar_data])
    ax.set_xticklabels([bd['label'] for bd in bar_data], rotation=45, ha='right')
    ax.set_ylabel('Δ Integral Activity (Bq·mL)', fontsize=12, fontweight='bold')
    ax.set_title('Waterfall Plot: Cumulative Change by Compartment\n(Red=Immune, Blue=Other)', 
                 fontsize=16, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, label='Immune (Increase)'),
        Patch(facecolor='#c0392b', alpha=0.8, label='Immune (Decrease)'),
        Patch(facecolor='#3498db', alpha=0.8, label='Other (Increase)'),
        Patch(facecolor='#2980b9', alpha=0.8, label='Other (Decrease)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "waterfall_compartment.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_immune_waterfall(df, out_dir):
    """
    Waterfall plot focused ONLY on immune compartments.
    Shows the overall immune system change broken down by category.
    """
    immune_df = df[df["Immune_Category"] != "Non-Immune"].copy()
    if len(immune_df) == 0:
        return
    
    metric = "Delta_Integral_Bq_mL"
    
    # Get category totals
    cat_totals = immune_df.groupby("Immune_Category")[metric].sum().sort_values(ascending=False)
    categories = list(cat_totals.index)
    values = cat_totals.values
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cumulative = 0
    bar_data = []
    
    # Starting point
    bar_data.append({
        'x': 0,
        'bottom': 0,
        'height': 0,
        'label': 'Baseline',
        'color': 'lightgray',
        'value': 0
    })
    
    # Color scheme for immune categories
    colors = {
        'Primary Immune': '#e74c3c',
        'Secondary Immune': '#e67e22',
        'Immune-Associated': '#f39c12'
    }
    
    # Each immune category
    for i, (cat, val) in enumerate(zip(categories, values)):
        if val >= 0:
            bar_data.append({
                'x': i + 1,
                'bottom': cumulative,
                'height': val,
                'label': cat,
                'color': colors.get(cat, '#95a5a6'),
                'value': val
            })
        else:
            bar_data.append({
                'x': i + 1,
                'bottom': cumulative + val,
                'height': abs(val),
                'label': cat,
                'color': colors.get(cat, '#95a5a6'),
                'value': val
            })
        
        cumulative += val
    
    # Ending point
    bar_data.append({
        'x': len(categories) + 1,
        'bottom': 0,
        'height': cumulative,
        'label': 'Net Immune\nChange',
        'color': '#2ecc71' if cumulative > 0 else '#c0392b',
        'value': cumulative
    })
    
    # Draw bars
    for bd in bar_data:
        bar = ax.bar(bd['x'], bd['height'], bottom=bd['bottom'], 
                     color=bd['color'], alpha=0.85, edgecolor='black', linewidth=2.5, width=0.7)
        
        # Add value labels
        label_y = bd['bottom'] + bd['height']/2
        if bd['height'] > 0:
            ax.text(bd['x'], label_y, f"{bd['value']:+.1f}", 
                    ha='center', va='center', fontweight='bold', fontsize=11,
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    
    # Draw connector lines
    for i in range(len(bar_data) - 1):
        x1, x2 = bar_data[i]['x'], bar_data[i+1]['x']
        y = bar_data[i]['bottom'] + (bar_data[i]['height'] if bar_data[i]['value'] >= 0 else 0)
        ax.plot([x1 + 0.35, x2 - 0.35], [y, y], 'k--', linewidth=2, alpha=0.7)
    
    # Formatting
    ax.axhline(0, color='black', linewidth=2.5)
    ax.set_xticks([bd['x'] for bd in bar_data])
    ax.set_xticklabels([bd['label'] for bd in bar_data], fontsize=11, fontweight='bold')
    ax.set_ylabel('Δ Integral Activity (Bq·mL)', fontsize=13, fontweight='bold')
    ax.set_title('Immune System Waterfall: Overall Change by Category', 
                 fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(out_dir, "waterfall_immune_overall.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_immune_component_waterfalls(df, out_dir):
    """
    Create separate waterfall plots for each immune category,
    breaking down into individual compartments.
    """
    immune_df = df[df["Immune_Category"] != "Non-Immune"].copy()
    if len(immune_df) == 0:
        return
    
    metric = "Delta_Integral_Bq_mL"
    categories = immune_df["Immune_Category"].unique()
    
    n_cats = len(categories)
    fig, axes = plt.subplots(n_cats, 1, figsize=(14, 5 * n_cats))
    if n_cats == 1:
        axes = [axes]
    
    fig.suptitle('Immune Component Waterfalls: Breakdown by Compartment', 
                 fontsize=18, fontweight='bold')
    
    colors_map = {
        'Primary Immune': ['#e74c3c', '#c0392b'],
        'Secondary Immune': ['#e67e22', '#d35400'],
        'Immune-Associated': ['#f39c12', '#e67e22']
    }
    
    for idx, category in enumerate(categories):
        ax = axes[idx]
        cat_data = immune_df[immune_df["Immune_Category"] == category]
        
        # Get compartment totals within this category
        comp_totals = cat_data.groupby("Compartment")[metric].sum().sort_values(ascending=False)
        compartments = list(comp_totals.index)
        values = comp_totals.values
        
        cumulative = 0
        bar_data = []
        
        # Starting point
        bar_data.append({
            'x': 0,
            'bottom': 0,
            'height': 0,
            'label': 'Start',
            'color': 'lightgray',
            'value': 0
        })
        
        # Each compartment
        base_color, dark_color = colors_map.get(category, ['#95a5a6', '#7f8c8d'])
        
        for i, (comp, val) in enumerate(zip(compartments, values)):
            if val >= 0:
                bar_data.append({
                    'x': i + 1,
                    'bottom': cumulative,
                    'height': val,
                    'label': comp,
                    'color': base_color,
                    'value': val
                })
            else:
                bar_data.append({
                    'x': i + 1,
                    'bottom': cumulative + val,
                    'height': abs(val),
                    'label': comp,
                    'color': dark_color,
                    'value': val
                })
            
            cumulative += val
        
        # Ending point
        bar_data.append({
            'x': len(compartments) + 1,
            'bottom': 0,
            'height': cumulative,
            'label': f'{category}\nTotal',
            'color': '#2ecc71' if cumulative > 0 else '#c0392b',
            'value': cumulative
        })
        
        # Draw bars
        for bd in bar_data:
            ax.bar(bd['x'], bd['height'], bottom=bd['bottom'], 
                   color=bd['color'], alpha=0.85, edgecolor='black', linewidth=2, width=0.7)
            
            # Add value labels
            if bd['height'] > 0:
                label_y = bd['bottom'] + bd['height']/2
                ax.text(bd['x'], label_y, f"{bd['value']:+.1f}", 
                        ha='center', va='center', fontweight='bold', fontsize=10,
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Draw connector lines
        for i in range(len(bar_data) - 1):
            x1, x2 = bar_data[i]['x'], bar_data[i+1]['x']
            y = bar_data[i]['bottom'] + (bar_data[i]['height'] if bar_data[i]['value'] >= 0 else 0)
            ax.plot([x1 + 0.35, x2 - 0.35], [y, y], 'k--', linewidth=1.5, alpha=0.6)
        
        # Formatting
        ax.axhline(0, color='black', linewidth=2)
        ax.set_xticks([bd['x'] for bd in bar_data])
        ax.set_xticklabels([bd['label'] for bd in bar_data], rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Δ Integral Activity (Bq·mL)', fontsize=11, fontweight='bold')
        ax.set_title(f'{category}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    path = os.path.join(out_dir, "waterfall_immune_components.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_total_activity_timeline(df, out_dir):
    """
    Simple before/after comparison showing total activity across all structures.
    """
    total_prev = df["Prev_Integral"].sum()
    total_curr = df["Curr_Integral"].sum()
    delta = total_curr - total_prev
    pct_change = (delta / total_prev * 100) if total_prev != 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    timepoints = ['Previous', 'Current']
    values = [total_prev, total_curr]
    colors = ['#95a5a6', '#2ecc71' if delta > 0 else '#e74c3c']
    
    bars = ax.bar(timepoints, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add change annotation
    ax.annotate('', xy=(1, total_curr), xytext=(0, total_prev),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    mid_y = (total_prev + total_curr) / 2
    ax.text(0.5, mid_y, f'{delta:+.1f}\n({pct_change:+.1f}%)',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Total Integral Activity (Bq·mL)', fontsize=12)
    ax.set_title('Overall Patient Activity: Before vs After', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "total_activity_timeline.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_compartment_timeline(df, out_dir):
    """
    Before/after comparison for each major compartment.
    """
    compartment_data = df.groupby("Compartment").agg({
        "Prev_Integral": "sum",
        "Curr_Integral": "sum",
        "Delta_Integral_Bq_mL": "sum"
    }).sort_values(by="Delta_Integral_Bq_mL", ascending=False)
    
    compartments = list(compartment_data.index)
    prev_vals = compartment_data["Prev_Integral"].values
    curr_vals = compartment_data["Curr_Integral"].values
    
    # Identify immune compartments
    immune_comps = [c for cat in IMMUNE_COMPARTMENTS.values() for c in cat]
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(compartments) * 0.5)))
    
    x = np.arange(len(compartments))
    width = 0.35
    
    # Color code: immune=red, other=blue
    prev_colors = ['#e74c3c' if c in immune_comps else '#3498db' for c in compartments]
    curr_colors = ['#c0392b' if c in immune_comps else '#2980b9' for c in compartments]
    
    bars1 = ax.barh(x - width/2, prev_vals, width, label='Previous', color=prev_colors, alpha=0.6)
    bars2 = ax.barh(x + width/2, curr_vals, width, label='Current', color=curr_colors, alpha=0.9)
    
    # Add value labels
    for i, (prev, curr) in enumerate(zip(prev_vals, curr_vals)):
        delta = curr - prev
        pct = (delta / prev * 100) if prev != 0 else 0
        ax.text(max(prev, curr) * 1.05, i, f'{delta:+.1f} ({pct:+.1f}%)',
                va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(x)
    ax.set_yticklabels([f"{c} *" if c in immune_comps else c for c in compartments])
    ax.set_xlabel('Integral Activity (Bq·mL)', fontsize=12)
    ax.set_title('Compartment Activity: Before vs After\n(* = Immune-related, Red=Immune, Blue=Other)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.axvline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    path = os.path.join(out_dir, "compartment_timeline.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)
    plt.close()

def create_summary_report(df, out_dir):
    """
    Generate comprehensive summary statistics and CSV reports.
    """
    # Overall summary
    total_prev = df["Prev_Integral"].sum()
    total_curr = df["Curr_Integral"].sum()
    total_delta = total_curr - total_prev
    total_pct = (total_delta / total_prev * 100) if total_prev != 0 else 0
    
    summary = {
        "Metric": ["Total Activity"],
        "Previous": [total_prev],
        "Current": [total_curr],
        "Delta": [total_delta],
        "% Change": [total_pct]
    }
    summary_df = pd.DataFrame(summary)
    
    # Compartment summary
    comp_summary = df.groupby("Compartment").agg({
        "Prev_Integral": "sum",
        "Curr_Integral": "sum",
        "Delta_Integral_Bq_mL": "sum"
    })
    comp_summary["% Change"] = (comp_summary["Curr_Integral"] - comp_summary["Prev_Integral"]) / comp_summary["Prev_Integral"] * 100
    comp_summary = comp_summary.sort_values("Delta_Integral_Bq_mL", ascending=False)
    
    # Immune summary
    immune_df = df[df["Immune_Category"] != "Non-Immune"]
    immune_summary = immune_df.groupby("Immune_Category").agg({
        "Prev_Integral": "sum",
        "Curr_Integral": "sum",
        "Delta_Integral_Bq_mL": "sum"
    })
    immune_summary["% Change"] = (immune_summary["Curr_Integral"] - immune_summary["Prev_Integral"]) / immune_summary["Prev_Integral"] * 100
    
    # Save all summaries
    summary_df.to_csv(os.path.join(out_dir, "summary_overall.csv"), index=False)
    comp_summary.to_csv(os.path.join(out_dir, "summary_by_compartment.csv"))
    immune_summary.to_csv(os.path.join(out_dir, "summary_immune_categories.csv"))
    
    print("Saved summary reports")
    
    # Print to console
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Total Previous Activity: {total_prev:.2f} Bq·mL")
    print(f"Total Current Activity:  {total_curr:.2f} Bq·mL")
    print(f"Total Change:            {total_delta:+.2f} Bq·mL ({total_pct:+.2f}%)")
    
    print("\n" + "="*60)
    print("TOP COMPARTMENT CHANGES")
    print("="*60)
    for comp, row in comp_summary.head(5).iterrows():
        print(f"{comp:30s}: {row['Delta_Integral_Bq_mL']:+10.2f} ({row['% Change']:+6.1f}%)")
    
    print("\n" + "="*60)
    print("IMMUNE SYSTEM CHANGES")
    print("="*60)
    for cat, row in immune_summary.iterrows():
        print(f"{cat:30s}: {row['Delta_Integral_Bq_mL']:+10.2f} ({row['% Change']:+6.1f}%)")
    print("="*60 + "\n")


# ==================== MAIN ENTRY POINT ====================
def compare_timepoints_interactive(curr_paths="generated_data", compare_data="generated_data2", 
                                 curr_pet_volume=None, structure_masks=None, structures=None, prev_pet_volume=None):
    """
    Main function to compare two timepoints and generate all visualizations.
    
    Parameters:
    - curr_paths: Path to current timepoint data (or dict with 'suv_stats' key)
    - compare_data: Path to previous timepoint data (or dict with 'suv_stats' key)
    - curr_pet_volume: Current PET volume for interactive viewer (optional)
    - prev_pet_volume: Previous PET volume for interactive viewer (optional)
    - structure_masks: Dictionary of structure masks (optional)
    - structures: Dictionary of structure metadata (optional)
    """
    
    # Load data
    try:
        curr_suv = load_timepoint_data(curr_paths)
    except Exception as e:
        try:
            print("Error", "Could not load current timepoint data:\n" + str(e))
        except Exception:
            print("[Compare] Could not load current timepoint data:", e)
        return

    try:
        prev_suv = load_timepoint_data(compare_data)
    except Exception as e:
        try:
            print("Error", "Could not load previous timepoint data:\n" + str(e))
        except Exception:
            print("[Compare] Could not load previous timepoint data:", e)
        return

    # Compute deltas
    df = compute_structure_deltas(curr_suv, prev_suv)
    out_dir = _ensure_compare_dir()
    
    # Save detailed CSV
    out_csv = os.path.join(out_dir, "per_structure_deltas.csv")
    df.to_csv(out_csv, index=False)
    print("Saved detailed delta CSV:", out_csv)

    # Generate all key visualizations
    print("\n[Compare] Generating visualizations...")
    
    # 1. Overall patient activity
    plot_total_activity_timeline(df, out_dir)
    
    # 2. Compartment-level changes
    plot_compartment_timeline(df, out_dir)
    # 2.5 Correlation: mean dose vs drop
    
    mean_dose_drop(df, out_dir)
    
    # 3. Waterfall plots - ALL THREE
    plot_waterfall_by_compartment(df, out_dir)  # All compartments
    plot_immune_waterfall(df, out_dir)  # Immune categories only
    plot_immune_component_waterfalls(df, out_dir)  # Each immune category breakdown
    
    # 4. Summary reports
    create_summary_report(df, out_dir)

    # Launch interactive viewer if volumes are provided
    from userInterface.gui import interactive_change_viewer
    try:
        if curr_pet_volume is not None and prev_pet_volume is not None and structure_masks is not None and structures is not None:
            print("[Compare] Launching interactive change viewer with hotspot detection...")
            interactive_change_viewer(curr_pet_volume, prev_pet_volume, structure_masks, structures, init_mode="absolute")
        else:
            print("[Compare] Skipping interactive viewer (volumes not provided).")
    except Exception as e_view:
        print("[Compare] Viewer error:", e_view)

    # Show completion message
    try:
        print("Analysis Complete", 
                          f"Change analysis completed successfully!\n\n"
                          f"Generated visualizations:\n"
                          f"• Overall patient activity timeline\n"
                          f"• Compartment before/after comparison\n"
                          f"• Waterfall: All compartments\n"
                          f"• Waterfall: Immune categories\n"
                          f"• Waterfall: Each immune component\n"
                          f"• Immune system overview (4 panels)\n"
                          f"• Immune compartment breakdown\n"
                          f"• Summary reports (CSV)\n\n"
                          f"All files saved to:\n{out_dir}")
    except Exception:
        print(f"\n[Compare] Analysis complete! All outputs saved to: {out_dir}")
        print("[Compare] Generated 8 key visualizations + summary reports")
