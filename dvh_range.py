import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d


def parse_dvh_wide_format(dvh_filepath):
    """
    Parse DVH file in wide format where:
    - Column 0: GY (dose values)
    - Columns 1+: Structure names with '_RelativeCumVolume' suffix
    
    Parameters:
    -----------
    dvh_filepath : str
        Path to the DVH CSV file
    
    Returns:
    --------
    tuple : (dose_values, dvh_dict)
        dose_values : numpy array of dose values in Gy
        dvh_dict : dictionary {structure_name: relative_volume_array}
    """
    print(f"Loading DVH file: {dvh_filepath}")
    
    # Load the file
    df = pd.read_csv(dvh_filepath)
    
    # First column should be dose (might be named 'GY' or similar)
    dose_column = df.columns[0]
    dose_values = df[dose_column].values
    
    print(f"  Dose column: '{dose_column}'")
    print(f"  Dose range: {np.min(dose_values):.3f} - {np.max(dose_values):.3f} Gy")
    print(f"  Number of dose points: {len(dose_values)}")
    
    # Parse structure columns
    # Format: "StructureName_RelativeCumVolume"
    dvh_dict = {}
    
    for col in df.columns[1:]:  # Skip dose column
        if '_RelativeCumVolume' in col:
            # Extract structure name by removing the suffix
            structure_name = col.replace('_RelativeCumVolume', '')
            dvh_dict[structure_name] = df[col].values
        else:
            # Column might be structure name directly
            structure_name = col
            dvh_dict[structure_name] = df[col].values
    
    print(f"  Found {len(dvh_dict)} structures")
    print(f"  Structure names: {list(dvh_dict.keys())}")
    
    return dose_values, dvh_dict


def get_volume_in_dose_range(dvh_curve, dose_values, dose_min=0, dose_max=3.0):
    """
    Calculate the fraction of volume receiving dose between dose_min and dose_max Gy
    from a cumulative DVH curve.
    
    Parameters:
    -----------
    dvh_curve : array-like
        Cumulative DVH values (fraction of volume receiving >= dose)
    dose_values : array-like
        Corresponding dose values in Gy
    dose_min : float
        Minimum dose in Gy (default: 2.0)
    dose_max : float
        Maximum dose in Gy (default: 3.0)
    
    Returns:
    --------
    float : Fraction of volume receiving between dose_min and dose_max Gy
    """
    # Remove NaN values
    valid_mask = ~np.isnan(dvh_curve) & ~np.isnan(dose_values)
    dvh_curve = np.array(dvh_curve)[valid_mask]
    dose_values = np.array(dose_values)[valid_mask]
    
    if len(dvh_curve) < 2:
        return np.nan
    
    # Ensure dose values are sorted
    sort_idx = np.argsort(dose_values)
    dose_values = dose_values[sort_idx]
    dvh_curve = dvh_curve[sort_idx]
    
    # Create interpolation function
    # Cumulative DVH: at dose D, value = fraction receiving >= D Gy
    interp_func = interp1d(dose_values, dvh_curve, kind='linear', 
                          bounds_error=False, fill_value=(dvh_curve[0], dvh_curve[-1]))
    
    # Get volume fractions
    vol_at_min = interp_func(dose_min)  # Fraction receiving >= dose_min Gy
    vol_at_max = interp_func(dose_max)  # Fraction receiving >= dose_max Gy
    
    # Volume receiving between dose_min and dose_max = difference
    vol_in_range = vol_at_min - vol_at_max
    
    return float(vol_in_range)


def match_structure_names(dvh_structures, delta_structures):
    """
    Create mapping between structure names in DVH file and delta CSV.
    Handles cases where names might have slight differences.
    
    Parameters:
    -----------
    dvh_structures : list
        Structure names from DVH file
    delta_structures : list
        Structure names from delta CSV
    
    Returns:
    --------
    dict : Mapping {delta_structure: dvh_structure}
    """
    mapping = {}
    
    for delta_struct in delta_structures:
        if delta_struct in dvh_structures:
            # Exact match
            mapping[delta_struct] = delta_struct
        else:
            # Try to find close match
            # Check with underscores replaced by spaces and vice versa
            delta_struct_alt = delta_struct.replace(' ', '_')
            if delta_struct_alt in dvh_structures:
                mapping[delta_struct] = delta_struct_alt
            else:
                # Check other way
                for dvh_struct in dvh_structures:
                    if dvh_struct.replace('_', ' ') == delta_struct:
                        mapping[delta_struct] = dvh_struct
                        break
    
    return mapping


def load_structure_volumes(generated_data_dir='generated_data', generated_data2_dir='generated_data2'):
    """
    Load absolute volumes for each structure.
    
    Returns:
    --------
    dict : Dictionary mapping structure names to absolute volumes in mL/cm³
    """
    volumes = {}
    
    # Try loading from a volumes file
    for data_dir in [generated_data_dir, generated_data2_dir]:
        volumes_path = os.path.join(data_dir, 'DVH/structure_volumes.csv')
        if os.path.exists(volumes_path):
            try:
                vol_df = pd.read_csv(volumes_path)
                if 'Structure' in vol_df.columns and 'Volume_mL' in vol_df.columns:
                    for _, row in vol_df.iterrows():
                        volumes[row['Structure']] = float(row['Volume_mL'])
                    print(f"✓ Loaded volumes from {volumes_path}")
                    return volumes
            except Exception as e:
                print(f"Error loading volumes from {volumes_path}: {e}")
    
    print("\n" + "="*80)
    print("ERROR: Structure volumes file not found!")
    print("="*80)
    print("Please create 'structure_volumes.csv' in your generated_data folder")
    print("Use the provided template: structure_volumes_TEMPLATE.csv")
    print("="*80 + "\n")
    
    return volumes


def dose_volume_vs_drop(df, output_dir='Comparisons', 
                       generated_data_dir='generated_data', 
                       generated_data2_dir='generated_data2',
                       dose_min=0, dose_max=3.0,
                       dvh_filename='CumulativeDVH_AllStructures_RelativeUnits.csv'):
    """
    Scatter plot showing whether absolute volume receiving 2-3 Gy correlates 
    with absolute drop in integral activity (immune PET).

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
    dose_min : float
        Minimum dose in Gy for volume calculation (default: 2.0)
    dose_max : float
        Maximum dose in Gy for volume calculation (default: 3.0)
    dvh_filename : str
        Name of the DVH curve file (default: 'CumulativeDVH_AllStructures_RelativeUnits.csv')
    """
    
    print("="*80)
    print("DOSE-VOLUME VS IMMUNE PET DROP ANALYSIS")
    print("="*80)
    print()
    
    # Load the delta data if not provided

    metric_y = 'Delta_Integral_Bq_mL'
    
    if metric_y not in df.columns:
        print(f"✗ Column '{metric_y}' not found in dataframe")
        print(f"  Available columns: {df.columns.tolist()}")
        return

    # Load structure volumes (absolute volumes in mL or cm³)
    print()
    structure_volumes = load_structure_volumes(generated_data_dir, generated_data2_dir)
    
    if not structure_volumes:
        return
    
    print(f"  Loaded {len(structure_volumes)} structure volumes")
    print()
    
    # Load DVH curves from generated_data (current scan)
    dvh_path1 = os.path.join(generated_data_dir, 'DVH', dvh_filename)
    
    if not os.path.exists(dvh_path1):
        print(f"✗ DVH file not found: {dvh_path1}")
        return
    
    dose_values1, dvh_dict1 = parse_dvh_wide_format(dvh_path1)
    print()
    
    # Optionally load from generated_data2 (previous scan) and average
    dvh_dict2 = None
    dose_values2 = None
    dvh_path2 = os.path.join(generated_data2_dir, 'DVH', dvh_filename)
    
    if os.path.exists(dvh_path2):
        try:
            dose_values2, dvh_dict2 = parse_dvh_wide_format(dvh_path2)
            print()
        except Exception as e:
            print(f"Warning: Could not load second DVH file: {e}")
            dvh_dict2 = None
    
    # Match structure names between files
    delta_structures = df['Structure'].unique().tolist()
    dvh_structures = list(dvh_dict1.keys())
    
    print("Matching structure names between files...")
    structure_mapping = match_structure_names(dvh_structures, delta_structures)
    print(f"  Matched {len(structure_mapping)} / {len(delta_structures)} structures")
    
    # Show unmatched structures
    unmatched = [s for s in delta_structures if s not in structure_mapping]
    if unmatched:
        print(f"  Unmatched structures: {unmatched[:5]}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched)-5} more")
    print()
    
    # Calculate absolute volume receiving dose_min to dose_max Gy for each structure
    print(f"Calculating volumes receiving {dose_min}-{dose_max} Gy...")
    volumes_in_range = {}
    
    for delta_struct in df['Structure'].unique():
        # Check if we have all needed data
        if delta_struct not in structure_mapping:
            continue
        
        dvh_struct = structure_mapping[delta_struct]
        
        if delta_struct not in structure_volumes:
            continue
        
        # Get relative volume fraction from DVH curve(s)
        rel_vol_frac1 = get_volume_in_dose_range(
            dvh_dict1[dvh_struct], 
            dose_values1, 
            dose_min, 
            dose_max
        )
        
        # If we have data from second timepoint, average them
        if dvh_dict2 is not None and dvh_struct in dvh_dict2:
            rel_vol_frac2 = get_volume_in_dose_range(
                dvh_dict2[dvh_struct],
                dose_values2,
                dose_min,
                dose_max
            )
            # Average the two fractions
            if not np.isnan(rel_vol_frac1) and not np.isnan(rel_vol_frac2):
                rel_vol_frac = (rel_vol_frac1 + rel_vol_frac2) / 2
            elif not np.isnan(rel_vol_frac1):
                rel_vol_frac = rel_vol_frac1
            else:
                rel_vol_frac = rel_vol_frac2
        else:
            rel_vol_frac = rel_vol_frac1
        
        if not np.isnan(rel_vol_frac) and rel_vol_frac >= 0:
            # Convert to absolute volume
            abs_volume = rel_vol_frac * structure_volumes[delta_struct]
            volumes_in_range[delta_struct] = abs_volume
            print(f"  {delta_struct:30s}: {rel_vol_frac:7.4f} × {structure_volumes[delta_struct]:8.1f} mL = {abs_volume:8.2f} mL")
    
    if not volumes_in_range:
        print("\n✗ No valid volume data calculated")
        print("  Check that structure names match between:")
        print("  - per_structure_deltas.csv")
        print("  - DVH file")
        print("  - structure_volumes.csv")
        return
    
    print(f"\n✓ Calculated volumes for {len(volumes_in_range)} structures")
    print()
    
    # Add volume data to dataframe
    df = df.copy()
    df['Volume_2_3Gy_mL'] = df['Structure'].map(volumes_in_range)
    
    # Drop rows without both volume and delta integral
    s = df.dropna(subset=['Volume_2_3Gy_mL', metric_y]).copy()
    
    # Optional: Filter by volume threshold
    min_volume_threshold = None  # Set to filter small volumes, e.g., 1.0 mL
    if min_volume_threshold is not None:
        n_before = len(s)
        s = s[s['Volume_2_3Gy_mL'] >= min_volume_threshold].copy()
        n_after = len(s)
        print(f"Filtered to volumes ≥ {min_volume_threshold} mL: {n_before} → {n_after} structures")
    
    if len(s) == 0:
        print("✗ No valid data points with both volume and delta integral")
        return
    
    print(f"✓ Plotting {len(s)} structures with complete data")
    print()
    
    # Prepare data for plotting
    x = s['Volume_2_3Gy_mL'].astype(float).values
    # If Delta_Integral_Bq_mL is negative for a drop, invert sign to make drops positive
    y = -s[metric_y].astype(float).values
    
    # Compute Pearson correlation and linear fit
    if len(x) > 1:
        try:
            pearson_r, pearson_p = stats.pearsonr(x, y)
            slope, intercept, r_value, p_value_lin, std_err = stats.linregress(x, y)
        except Exception:
            pearson_r = np.corrcoef(x, y)[0, 1]
            pearson_p = np.nan
            slope, intercept = np.polyfit(x, y, 1)
            r_value = pearson_r
            p_value_lin = np.nan
            std_err = np.nan
    else:
        pearson_r = np.nan
        pearson_p = np.nan
        slope = 0.0
        intercept = np.mean(y) if len(y) > 0 else 0.0
        r_value = np.nan
        p_value_lin = np.nan
        std_err = np.nan
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(11, 8))
    
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
                ax.scatter(x[mask], y[mask], s=100, c=color, edgecolor='black', 
                          alpha=0.85, label=category, linewidth=1.5)
        # Add any uncategorized points
        mask = ~s['Immune_Category'].isin(cmap.keys())
        if mask.any():
            ax.scatter(x[mask], y[mask], s=100, c='#95a5a6', edgecolor='black', 
                      alpha=0.85, label='Other', linewidth=1.5)
    else:
        ax.scatter(x, y, s=100, c='#2c7fb8', edgecolor='black', alpha=0.85, linewidth=1.5)
    
    # Add regression line
    if not np.isnan(slope) and len(x) > 1:
        xs = np.linspace(np.min(x), np.max(x), 200)
        ys = slope * xs + intercept
        ax.plot(xs, ys, color='#f03b20', lw=3, linestyle='--', 
               label=f'Linear fit (slope={slope:.2e})', zorder=10)
    
    # Labels and title
    ax.set_xlabel(f'Absolute Volume Receiving {dose_min}-{dose_max} Gy (mL)', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Absolute Drop in Immune PET Activity\n(Integral Activity, Bq·mL)', 
                  fontsize=14, fontweight='bold')
    ax.set_title(f'Volume Receiving {dose_min}-{dose_max} Gy vs Drop in Immune PET Activity', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Annotate correlation info
    ann_x = 0.05
    ann_y = 0.95
    n_pts = len(x)
    
    if not np.isnan(pearson_r):
        ann_text = f'n = {n_pts}\nPearson r = {pearson_r:.3f}'
        if not np.isnan(pearson_p):
            if pearson_p < 0.001:
                ann_text += f'\np < 0.001'
            elif pearson_p < 0.01:
                ann_text += f'\np < 0.01'
            else:
                ann_text += f'\np = {pearson_p:.4f}'
        else:
            ann_text += '\np = n/a'
        
        # Add R² if available
        if not np.isnan(r_value):
            ann_text += f'\nR² = {r_value**2:.3f}'
    else:
        ann_text = f'n = {n_pts}'
    
    ax.text(ann_x, ann_y, ann_text,
            transform=ax.transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=2))
    
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    
    if 'Immune_Category' in s.columns or not np.isnan(slope):
        ax.legend(loc='best', framealpha=0.95, edgecolor='gray', fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    output_filename = f'volume_{dose_min}_{dose_max}Gy_vs_drop.png'
    path = os.path.join(output_dir, output_filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f'✓ Saved plot: {path}')
    plt.close()
    
    # Print summary statistics
    print()
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Volume range ({dose_min}-{dose_max} Gy): {np.min(x):.2f} - {np.max(x):.2f} mL")
    print(f"  Mean: {np.mean(x):.2f} mL")
    print(f"  Median: {np.median(x):.2f} mL")
    print()
    print(f"Drop range: {np.min(y):.2f} - {np.max(y):.2f} Bq·mL")
    print(f"  Mean: {np.mean(y):.2f} Bq·mL")
    print(f"  Median: {np.median(y):.2f} Bq·mL")
    print()
    if not np.isnan(pearson_r):
        print(f"Correlation: r = {pearson_r:.4f}")
        if not np.isnan(pearson_p):
            print(f"  p-value = {pearson_p:.6f}")
            if pearson_p < 0.05:
                print(f"  *** Statistically significant (p < 0.05) ***")
        if not np.isnan(r_value):
            print(f"  R² = {r_value**2:.4f}")
        if not np.isnan(slope):
            print(f"\nLinear fit equation:")
            print(f"  Drop = {slope:.4e} × Volume + {intercept:.2f}")
    print("="*80)
    print()
    
    # Save data to CSV for further analysis
    output_csv = os.path.join(output_dir, f'volume_{dose_min}_{dose_max}Gy_vs_drop_data.csv')
    output_cols = ['Structure', 'Volume_2_3Gy_mL', metric_y]
    if 'Immune_Category' in s.columns:
        output_cols.insert(1, 'Immune_Category')
    s[output_cols].to_csv(output_csv, index=False)
    print(f"✓ Saved data to: {output_csv}")
    print()
    
    return s


# Example usage:
if __name__ == "__main__":
    # Run the analysis
    result = dose_volume_vs_drop(
        df=pd.read_csv(os.path.join('Comparisons', 'per_structure_deltas.csv')),  # Will load from per_structure_deltas.csv
        output_dir='Comparisons',
        generated_data_dir='generated_data',
        generated_data2_dir='generated_data2',
        dose_min=0,
        dose_max=3.0,
        dvh_filename='CumulativeDVH_AllStructures_RelativeUnits.csv'
    )
