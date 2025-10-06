import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button
from matplotlib.patches import Circle
from scipy import ndimage
from skimage.measure import label, regionprops

# ---------------- Interactive change viewer with hotspots ----------------
def interactive_change_viewer(curr_pet_volume, prev_pet_volume, structure_masks, structures,
                              init_mode="absolute"):
    """
    Interactive viewer for changes in PET activity between two 3D stacks with hotspot detection.

    - curr_pet_volume, prev_pet_volume: np.ndarray [Z, Y, X], same shape
    - structure_masks: dict {sid: 3D bool array aligned to current volume}
    - structures: dict {sid: {"name": str, ...}}
    """

    if curr_pet_volume is None or prev_pet_volume is None:
        print("[Viewer] Need both current and previous PET volumes; skipping.")
        return
    if curr_pet_volume.shape != prev_pet_volume.shape:
        print("[Viewer] Volumes have different shapes; skipping.")
        return
    if not structure_masks:
        print("[Viewer] No structure masks provided; skipping.")
        return

    # Compute change arrays
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = curr_pet_volume.astype(np.float32) - prev_pet_volume.astype(np.float32)
        pct   = np.where(prev_pet_volume != 0,
                         (delta / prev_pet_volume) * 100.0,
                         0.0).astype(np.float32)

    mode_names = ["absolute", "percent"]
    mode_idx = 0 if init_mode not in mode_names else mode_names.index(init_mode)
    change_stack = {"absolute": delta, "percent": pct}

    struct_ids   = list(structure_masks.keys())
    struct_names = [structures[sid]['name'] for sid in struct_ids]
    selected_structs = [struct_ids[0], struct_ids[1] if len(struct_ids) > 1 else struct_ids[0]]

    z_slices = curr_pet_volume.shape[0]
    slice_idx = z_slices // 2

    diverging_maps = ['coolwarm', 'seismic', 'bwr', 'PiYG', 'PRGn']
    cmap_name = diverging_maps[0]

    # Hotspot detection parameters
    show_hotspots = True
    hotspot_threshold_percentile = 95
    min_hotspot_size = 5

    def _robust_symmetric_clim(arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return (-1.0, 1.0)
        lo, hi = np.percentile(finite, [2, 98])
        m = float(max(abs(lo), abs(hi)))
        return (-m if m else -1.0, m if m else 1.0)

    def _detect_hotspots(change_slice, threshold_percentile=95, min_size=5):
        """Detect hotspots of activity change in a 2D slice."""
        finite_values = change_slice[np.isfinite(change_slice)]
        if len(finite_values) == 0:
            return [], []
        
        # Separate positive and negative changes
        pos_changes = finite_values[finite_values > 0]
        neg_changes = finite_values[finite_values < 0]
        
        hotspots_increase = []
        hotspots_decrease = []
        
        # Detect increase hotspots
        if len(pos_changes) > 0:
            pos_threshold = np.percentile(pos_changes, threshold_percentile)
            pos_mask = (change_slice >= pos_threshold) & np.isfinite(change_slice)
            labeled_pos = label(pos_mask)
            
            for region in regionprops(labeled_pos):
                if region.area >= min_size:
                    y, x = region.centroid
                    intensity = np.mean(change_slice[labeled_pos == region.label])
                    hotspots_increase.append((x, y, intensity, region.area))
        
        # Detect decrease hotspots
        if len(neg_changes) > 0:
            neg_threshold = np.percentile(neg_changes, 100 - threshold_percentile)
            neg_mask = (change_slice <= neg_threshold) & np.isfinite(change_slice)
            labeled_neg = label(neg_mask)
            
            for region in regionprops(labeled_neg):
                if region.area >= min_size:
                    y, x = region.centroid
                    intensity = np.mean(change_slice[labeled_neg == region.label])
                    hotspots_decrease.append((x, y, intensity, region.area))
        
        return hotspots_increase, hotspots_decrease

    clim = {k: _robust_symmetric_clim(v) for k, v in change_stack.items()}

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.35, bottom=0.35)

    img = ax.imshow(
        change_stack[mode_names[mode_idx]][slice_idx],
        cmap=cmap_name,
        origin='lower',
        vmin=clim[mode_names[mode_idx]][0],
        vmax=clim[mode_names[mode_idx]][1]
    )
    ax.set_title(f"Slice {slice_idx} — Change ({mode_names[mode_idx]})")

    # Widgets
    ax_slice = plt.axes([0.35, 0.25, 0.6, 0.03])
    slice_slider = Slider(ax_slice, 'Slice', 0, z_slices - 1, valinit=slice_idx, valfmt='%0.0f')

    ax_struct = plt.axes([0.025, 0.5, 0.25, 0.4], frameon=False)
    struct_check = CheckButtons(ax_struct, struct_names, [sid in selected_structs for sid in struct_ids])

    ax_cmap = plt.axes([0.025, 0.35, 0.25, 0.12], frameon=False)
    cmap_radio = RadioButtons(ax_cmap, diverging_maps, active=diverging_maps.index(cmap_name))

    ax_mode = plt.axes([0.025, 0.25, 0.25, 0.08], frameon=False)
    mode_radio = RadioButtons(ax_mode, mode_names, active=mode_idx)

    # Hotspot controls
    ax_hotspot_btn = plt.axes([0.025, 0.15, 0.12, 0.04])
    hotspot_btn = Button(ax_hotspot_btn, 'Toggle Hotspots')

    ax_threshold = plt.axes([0.15, 0.15, 0.15, 0.03])
    threshold_slider = Slider(ax_threshold, 'Threshold %', 90, 99, valinit=hotspot_threshold_percentile, valfmt='%0.0f')

    ax_min_size = plt.axes([0.025, 0.1, 0.25, 0.03])
    size_slider = Slider(ax_min_size, 'Min Size', 1, 20, valinit=min_hotspot_size, valfmt='%0.0f')

    def draw_overlays(z):
        # Structure overlays - alternate red/lime for the two selected overlays
        for idx, sid in enumerate(struct_ids):
            if sid in selected_structs:
                color = 'red' if idx % 2 == 0 else 'lime'
                m = structure_masks[sid]
                if m.shape[0] != z_slices:
                    continue
                ax.contour(m[z].astype(bool), colors=color, linewidths=1.5)

    def draw_hotspots(z, mode_key):
        if not show_hotspots:
            return
        
        change_slice = change_stack[mode_key][z]
        hotspots_inc, hotspots_dec = _detect_hotspots(
            change_slice, 
            threshold_percentile=threshold_slider.val,
            min_size=int(size_slider.val)
        )
        
        # Draw increase hotspots (red circles)
        for x, y, intensity, area in hotspots_inc:
            circle = Circle((x, y), radius=np.sqrt(area/np.pi), 
                          fill=False, color='orange', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, f'+{intensity:.1f}', ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=8)
        
        # Draw decrease hotspots (blue circles)
        for x, y, intensity, area in hotspots_dec:
            circle = Circle((x, y), radius=np.sqrt(area/np.pi), 
                          fill=False, color='cyan', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, f'{intensity:.1f}', ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=8)

    def update_display(z, mode_key=None):
        if mode_key is None:
            mode_key = mode_radio.value_selected
        arr = change_stack[mode_key]
        ax.clear()
        ax.imshow(
            arr[z],
            cmap=img.get_cmap(),
            origin='lower',
            vmin=clim[mode_key][0],
            vmax=clim[mode_key][1]
        )
        draw_overlays(z)
        draw_hotspots(z, mode_key)
        
        hotspot_status = "ON" if show_hotspots else "OFF"
        ax.set_title(f"Slice {z} — Change ({mode_key}) — Hotspots: {hotspot_status}\n"
                     f"Viewing: {', '.join(structures[sid]['name'] for sid in selected_structs)}")
        fig.canvas.draw_idle()

    def on_slice_change(_val):
        update_display(int(slice_slider.val))

    def on_structure_toggle(label):
        idx = struct_names.index(label)
        sid = struct_ids[idx]
        if sid in selected_structs:
            selected_structs.remove(sid)
        else:
            if len(selected_structs) >= 2:
                selected_structs.pop(0)
            selected_structs.append(sid)
        update_display(int(slice_slider.val))

    def on_cmap_change(label):
        img.set_cmap(label)
        update_display(int(slice_slider.val))

    def on_mode_change(label):
        update_display(int(slice_slider.val), mode_key=label)

    def on_hotspot_toggle(_event):
        nonlocal show_hotspots
        show_hotspots = not show_hotspots
        update_display(int(slice_slider.val))

    def on_threshold_change(_val):
        update_display(int(slice_slider.val))

    def on_size_change(_val):
        update_display(int(slice_slider.val))

    slice_slider.on_changed(on_slice_change)
    struct_check.on_clicked(on_structure_toggle)
    cmap_radio.on_clicked(on_cmap_change)
    mode_radio.on_clicked(on_mode_change)
    hotspot_btn.on_clicked(on_hotspot_toggle)
    threshold_slider.on_changed(on_threshold_change)
    size_slider.on_changed(on_size_change)

    def on_key(event):
        z = int(slice_slider.val)
        out_dir = os.path.join("generated_data", "Comparisons", "Images")
        if event.key == 's':
            os.makedirs(out_dir, exist_ok=True)
            filename = os.path.join(out_dir, f"Slice_{z:03d}_Change_{mode_radio.value_selected}.png")
            fig.savefig(filename, dpi=600, bbox_inches='tight')
            print(f"[Viewer] Saved: {filename}")
        elif event.key == 'right' and z < z_slices - 1:
            slice_slider.set_val(z + 1)
        elif event.key == 'left' and z > 0:
            slice_slider.set_val(z - 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update_display(slice_idx, mode_key=mode_names[mode_idx])
    plt.show()


# ---------------- Enhanced Compartments with Immune Focus ----------------
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

# Define immune-related compartments for special handling
IMMUNE_COMPARTMENTS = {
    "Primary Immune Organs": ["Bone Marrow", "Thymus"],
    "Secondary Immune Organs": ["Spleen", "Lymph Nodes"],
    "Immune-Associated Organs": ["Liver", "Lungs", "Bowel/GI"]
}

# ---------------- Helpers ----------------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing file: " + path)
    
    # Try different encodings to handle special characters
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            df.columns = [col.replace('Â', '') for col in df.columns]
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if encoding == encodings[-1]: 
                raise e
            continue
    
    try:
        return pd.read_csv(path, encoding='utf-8', errors='replace')
    except Exception as e:
        raise Exception(f"Could not read CSV file {path} with any encoding: {str(e)}")

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
    """Assign immune system category to compartments."""
    for category, comps in IMMUNE_COMPARTMENTS.items():
        if compartment in comps:
            return category
    return "Non-Immune"

def _resolve_timepoint_paths(paths_like):
    """
    Accepts either:
      - dict with key 'suv_stats', or
      - a directory string pointing to either the analysis ROOT or the 'generated_data' folder.

    Returns dict {'suv_stats': <path>}
    """
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

# ---------------- Delta computation ----------------
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

    out["Delta_Volume_mL"]   = merged["Volume (mL)_curr"]    - merged["Volume (mL)_prev"]
    out["Delta_SUVmean"]     = merged["SUVmean_curr"]        - merged["SUVmean_prev"]
    out["Delta_SUVmax"]      = merged["SUVmax_curr"]         - merged["SUVmax_prev"]
    out["Delta_Integral_Bq_mL"] = merged[f"{integral_col_curr}_curr"] - merged[f"{integral_col_prev}_prev"]

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

# ---------------- Enhanced Plots with Immune Focus ----------------
def _ensure_compare_dir():
    out_dir = os.path.join("generated_data", "Comparisons")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def plot_immune_compartment_stacked_changes(df, out_dir):
    """Create stacked bar charts focusing on immune compartments."""
    metric = "Delta_Integral_Bq_mL"
    s = df.dropna(subset=[metric])
    if len(s) == 0:
        return

    # Create immune system overview
    immune_data = s[s["Immune_Category"] != "Non-Immune"].copy()
    if len(immune_data) == 0:
        return

    # Group by immune category and compartment
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Stacked by Immune Category
    category_totals = immune_data.groupby("Immune_Category")[metric].sum().sort_values(ascending=True)
    categories = list(category_totals.index)
    
    # Get color map
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    # Create stacked bars by structure within each category
    bottom_values = np.zeros(len(categories))
    structure_contributions = {}
    
    for cat_idx, category in enumerate(categories):
        cat_data = immune_data[immune_data["Immune_Category"] == category].sort_values(by=metric, ascending=False)
        
        # Get top contributing structures
        top_structures = cat_data.head(8)  # Top 8 structures per category
        
        for struct_idx, (_, row) in enumerate(top_structures.iterrows()):
            structure_name = row["Structure"]
            value = row[metric]
            
            if structure_name not in structure_contributions:
                structure_contributions[structure_name] = [0.0] * len(categories)
            structure_contributions[structure_name][cat_idx] = value

    # Plot stacked bars
    bottom = np.zeros(len(categories))
    structure_colors = plt.cm.tab20(np.linspace(0, 1, len(structure_contributions)))
    
    for idx, (structure, values) in enumerate(structure_contributions.items()):
        values_array = np.array(values)
        ax1.barh(categories, values_array, left=bottom, 
                label=structure, color=structure_colors[idx % len(structure_colors)])
        bottom += values_array

    ax1.set_title("Immune System Changes: Δ Integral Activity by Category (Stacked by Structure)")
    ax1.set_xlabel("Delta Integral Activity (Bq·mL)")
    ax1.axvline(0, color='black', linewidth=1, linestyle='--')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

    # Plot 2: Compartment-level stacked view
    compartment_totals = immune_data.groupby("Compartment")[metric].sum().sort_values(ascending=True)
    compartments = list(compartment_totals.index)
    
    compartment_contributions = {}
    for comp in compartments:
        comp_data = immune_data[immune_data["Compartment"] == comp].sort_values(by=metric, ascending=False)
        
        for _, row in comp_data.iterrows():
            structure_name = row["Structure"]
            value = row[metric]
            
            if structure_name not in compartment_contributions:
                compartment_contributions[structure_name] = [0.0] * len(compartments)
            comp_idx = compartments.index(comp)
            compartment_contributions[structure_name][comp_idx] = value

    # Plot compartment stacked bars
    bottom = np.zeros(len(compartments))
    for idx, (structure, values) in enumerate(compartment_contributions.items()):
        values_array = np.array(values)
        ax2.barh(compartments, values_array, left=bottom, 
                label=structure, color=structure_colors[idx % len(structure_colors)])
        bottom += values_array

    ax2.set_title("Immune-Related Compartments: Δ Integral Activity (Stacked by Structure)")
    ax2.set_xlabel("Delta Integral Activity (Bq·mL)")
    ax2.axvline(0, color='black', linewidth=1, linestyle='--')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    path = os.path.join(out_dir, "immune_compartment_stacked_changes.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)
    plt.close()

def plot_per_structure_changes(df, out_dir):
    """Enhanced structure plots with immune highlighting."""
    # Separate immune and non-immune structures
    immune_df = df[df["Immune_Category"] != "Non-Immune"]
    
    metrics = [
        ("Delta_Integral_Bq_mL", "Change in Integral Activity by Structure", "per_structure_delta_integral.png"),
        ("Delta_SUVmean", "Change in SUVmean by Structure", "per_structure_delta_suvmean.png"),
        ("Delta_SUVmax", "Change in SUVmax by Structure", "per_structure_delta_suvmax.png"),
        ("Delta_Volume_mL", "Change in Volume (mL) by Structure", "per_structure_delta_volume.png")
    ]
    
    for col, title, filename in metrics:
        s = df.dropna(subset=[col]).copy()
        if len(s) == 0:
            continue
            
        plt.figure(figsize=(12, max(4, 0.35 * len(s))))
        
        # Color code bars: immune structures in red, others in blue
        colors = ['red' if cat != "Non-Immune" else 'lightblue' for cat in s["Immune_Category"]]
        
        bars = plt.barh(s["Structure"], s[col], color=colors, alpha=0.7)
        plt.axvline(0, color='black', linewidth=1)
        plt.title(f"{title} (Red = Immune-Related)")
        plt.xlabel(col)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Immune-Related'),
                          Patch(facecolor='lightblue', alpha=0.7, label='Other')]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print("Saved:", path)
        plt.close()

def plot_compartment_contributions(df, out_dir):
    """Enhanced compartment plot with immune focus."""
    plot_immune_compartment_stacked_changes(df, out_dir)
    
    # Also create the original compartment plot
    metric = "Delta_Integral_Bq_mL"
    s = df.dropna(subset=[metric])
    if len(s) == 0:
        return

    totals = s.groupby("Compartment")[metric].sum().sort_values(ascending=False)
    compartments = list(totals.index)

    stacks = {}
    for comp in compartments:
        sub = s[s["Compartment"] == comp].sort_values(by=metric, ascending=False)
        for _, row in sub.iterrows():
            name = row["Structure"]
            if name not in stacks:
                stacks[name] = [0.0] * len(compartments)
            idx = compartments.index(comp)
            stacks[name][idx] = float(row[metric]) if pd.notna(row[metric]) else 0.0

    def magnitude(vals): return sum(abs(x) for x in vals)
    top_names = sorted(stacks.keys(), key=lambda k: magnitude(stacks[k]), reverse=True)[:15]

    plt.figure(figsize=(12, max(4, 0.6 * len(compartments))))
    bottom = np.zeros(len(compartments))
    
    # Color immune-related compartments differently
    immune_compartment_names = []
    for cat_comps in IMMUNE_COMPARTMENTS.values():
        immune_compartment_names.extend(cat_comps)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_names)))
    
    for idx, name in enumerate(top_names):
        vals = np.array(stacks[name], dtype=float)
        plt.barh(compartments, vals, left=bottom, label=name, color=colors[idx])
        bottom = bottom + vals

    plt.title("All Compartments: Δ Integral Activity (Stacked by Structure)")
    plt.xlabel("Delta Integral (Bq·mL)")
    
    # Highlight immune compartments in y-axis labels
    y_labels = []
    for comp in compartments:
        if comp in immune_compartment_names:
            y_labels.append(f"{comp} *")
        else:
            y_labels.append(comp)
    plt.gca().set_yticklabels(y_labels)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.figtext(0.02, 0.02, "* = Immune-related compartments", fontsize=10, style='italic')
    plt.tight_layout()
    path = os.path.join(out_dir, "compartment_contributions_integral.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)
    plt.close()

def create_immune_summary_report(df, out_dir):
    """Create a summary report focusing on immune system changes."""
    immune_df = df[df["Immune_Category"] != "Non-Immune"].copy()
    if len(immune_df) == 0:
        return
    
    # Calculate summary statistics
    summary_stats = {}
    for category in IMMUNE_COMPARTMENTS.keys():
        cat_data = immune_df[immune_df["Immune_Category"] == category]
        if len(cat_data) > 0:
            summary_stats[category] = {
                "total_structures": len(cat_data),
                "total_delta_integral": cat_data["Delta_Integral_Bq_mL"].sum(),
                "avg_delta_integral": cat_data["Delta_Integral_Bq_mL"].mean(),
                "max_increase": cat_data["Delta_Integral_Bq_mL"].max(),
                "max_decrease": cat_data["Delta_Integral_Bq_mL"].min(),
                "structures_increased": len(cat_data[cat_data["Delta_Integral_Bq_mL"] > 0]),
                "structures_decreased": len(cat_data[cat_data["Delta_Integral_Bq_mL"] < 0])
            }
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_stats).T
    summary_path = os.path.join(out_dir, "immune_system_summary.csv")
    summary_df.to_csv(summary_path)
    print("Saved immune summary:", summary_path)
    
    # Create detailed immune structures CSV
    immune_detailed_path = os.path.join(out_dir, "immune_structures_detailed.csv")
    immune_df.to_csv(immune_detailed_path, index=False)
    print("Saved detailed immune data:", immune_detailed_path)

# ---------------- entrypoint ----------------
def compare_timepoints_interactive(curr_paths="generated_data", compare_data="generated_data2", 
                                 curr_pet_volume=None, structure_masks=None, structures=None, prev_pet_volume=None):
    """
    curr_paths: dict OR path to either ./generated_data or the analysis root.
    compare_data: dict OR path to other timepoint's generated_data or its root.
    """
    try:
        curr_suv = load_timepoint_data(curr_paths)
    except Exception as e:
        try:
            messagebox.showerror("Error", "Could not load current timepoint data:\n" + str(e))
        except Exception:
            print("[Compare] Could not load current timepoint data:", e)
        return

    try:
        prev_suv = load_timepoint_data(compare_data)
    except Exception as e:
        try:
            messagebox.showerror("Error", "Could not load previous timepoint data:\n" + str(e))
        except Exception:
            print("[Compare] Could not load previous timepoint data:", e)
        return

    df = compute_structure_deltas(curr_suv, prev_suv)
    out_dir = _ensure_compare_dir()
    out_csv = os.path.join(out_dir, "per_structure_deltas.csv")
    df.to_csv(out_csv, index=False)
    print("Delta per-structure CSV:", out_csv)

    # Generate enhanced plots
    plot_per_structure_changes(df, out_dir)
    plot_compartment_contributions(df, out_dir)
    
    # Create immune system focused analysis
    create_immune_summary_report(df, out_dir)

    # Launch the interactive change viewer with hotspot detection
    try:
        if curr_pet_volume is not None and prev_pet_volume is not None and structure_masks is not None and structures is not None:
            print("[Compare] Launching enhanced interactive change viewer with hotspot detection...")
            interactive_change_viewer(curr_pet_volume, prev_pet_volume, structure_masks, structures, init_mode="absolute")
        else:
            print("[Compare] Skipping interactive viewer (needs curr, prev, masks, and structures).")
    except Exception as _e_view:
        print("[Compare] Viewer error:", _e_view)

    try:
        messagebox.showinfo("Done", "Enhanced change analysis completed.\nSaved outputs in:\n" + out_dir + 
                          "\n\nNew features:\n- Immune compartment stacked charts\n- Hotspot detection in viewer\n- Immune system summary report")
    except Exception:
        print("[Compare] Enhanced change analysis saved to:", out_dir)
        print("[Compare] New features: immune compartment analysis and hotspot detection")