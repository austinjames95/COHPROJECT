import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from config import given_pvh_dir, given_dvh_dir

def normalize_structure_name(name):
    """Strip outer whitespace and replace internal spaces with underscores, keeping dashes."""
    if not isinstance(name, str):
        name = str(name)

    # Strip leading/trailing whitespace
    name = name.strip()

    # Replace only spaces/tabs/newlines with underscore, not hyphens
    name = name.replace(" ", "_")

    return name

def compare_relative_dvh():
    if given_dvh_dir is None:
        return
    df = pd.read_csv(given_dvh_dir, header=None)

    blank_indices = df.index[df.isnull().all(axis=1)].tolist()

    if 0 not in blank_indices:
        blank_indices = [0] + blank_indices

    # Add the start (0) and end (len(df)) to help split
    split_points = [0] + blank_indices + [len(df)]

    # Split into chunks between blank rows
    sections = []

    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chunk = df.iloc[start:end].dropna(how='all')  # Drop any full-NaN rows just in case
        if not chunk.empty:
            sections.append(chunk.reset_index(drop=True))

    cdvh_results = {}

    for section in sections:
        if section.shape[0] < 3:
            continue  

        structure_name = str(section.iloc[0, 0]).strip()
        
        dose_row_index = section[section[0].astype(str).str.upper() == "GY"].index
        if dose_row_index.empty:
            continue  
        
        header_idx = dose_row_index[0]
        data_rows = section.iloc[header_idx + 1:]
        data_rows = data_rows.iloc[:, :2]  
        data_rows.columns = ["GY", "cm3"]
        data_rows = data_rows.dropna()
        
        try:
            data_rows["GY"] = pd.to_numeric(data_rows["GY"], errors='coerce')
            data_rows["cm3"] = pd.to_numeric(data_rows["cm3"], errors='coerce')
            data_rows = data_rows.dropna()
            data_rows = data_rows.sort_values(by="GY").reset_index(drop=True)
            total_volume = data_rows["cm3"].sum()
            data_rows["CumulativeVolume_cm3"] = data_rows["cm3"][::-1].cumsum()[::-1]
            data_rows["RelativeCumVolume"] = data_rows["CumulativeVolume_cm3"] / total_volume

            cdvh_results[structure_name] = data_rows

        except Exception as e:
            print(f"Skipping {structure_name} due to error: {e}")

    for name, cdvh in cdvh_results.items():
        plt.plot(cdvh["GY"], cdvh["RelativeCumVolume"], label=name)

    plt.xlabel("Dose (Gy)", fontsize=12)
    plt.ylabel("Relative Volume (fraction)", fontsize=12)
    plt.title("Cumulative DVH - Relative Volumes", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    rel_plot_path = os.path.join("generated_data/DVH", "CDVH_Converted.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')

    print(f" Relative DVH plot saved to: {rel_plot_path}")

    try:
        ref_path = os.path.join("generated_data/DVH", "CumulativeDVH_AllStructures_RelativeUnits.csv")
        ref_df = pd.read_csv(ref_path, comment='#')
        dose_ref = ref_df["GY"].values

        diff_output = {"GY": dose_ref}

        for name, cdvh in cdvh_results.items():
            clean_name = normalize_structure_name(name)
            rel_col = f"{clean_name}_RelativeCumVolume"
            if rel_col not in ref_df.columns:
                print(f"Structure '{name}' not found in DICOM DVH. Skipping diff.")
                continue
            interp = np.interp(dose_ref, cdvh["GY"], cdvh["RelativeCumVolume"], left=np.nan, right=np.nan)
            ref_values = ref_df[rel_col].values
            difference = interp - ref_values

            diff_output[f"{name}_Diff"] = difference

        # Save the difference as a CSV
        diff_df = pd.DataFrame(diff_output)
        diff_path = os.path.join("generated_data/DVH", "DVH_Differences.csv")
        diff_df.to_csv(diff_path, index=False)
        print(f"DVH differences saved to: {diff_path}")

    except Exception as e:
        print(f"Error comparing to reference DVH: {e}")

def compare_relative_pvh():
    if given_pvh_dir is None:
        return
    df = pd.read_csv(given_pvh_dir, header=None)

    blank_indices = df.index[df.isnull().all(axis=1)].tolist()

    if 0 not in blank_indices:
        blank_indices = [0] + blank_indices


    split_points = blank_indices + [len(df)]

    sections = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chunk = df.iloc[start:end].dropna(how='all')
        if not chunk.empty:
            sections.append(chunk.reset_index(drop=True))

    pvh_results = {}

    for section in sections:
        if section.shape[0] < 3:
            continue  

        structure_name = str(section.iloc[0, 0]).strip()
        
        value_row_index = section[section[0].astype(str).str.upper().str.contains("BQ|SUV|PET", na=False)].index
        if value_row_index.empty:
            continue

        header_idx = value_row_index[0]
        data_rows = section.iloc[header_idx + 1:]
        data_rows = data_rows.iloc[:, :2]  
        data_rows.columns = ["BQML", "cm3"]
        data_rows = data_rows.dropna()

        try:
            data_rows["BQML"] = pd.to_numeric(data_rows["BQML"], errors='coerce')
            data_rows["cm3"] = pd.to_numeric(data_rows["cm3"], errors='coerce')
            data_rows = data_rows.dropna()

            data_rows = data_rows.sort_values(by="BQML").reset_index(drop=True)
            total_volume = data_rows["cm3"].sum()

            data_rows["CumulativeVolume_mL"] = data_rows["cm3"][::-1].cumsum()[::-1]
            data_rows["RelativeCumVolume"] = data_rows["CumulativeVolume_mL"] / total_volume

            pvh_results[structure_name] = data_rows

        except Exception as e:
            print(f"Skipping {structure_name} due to error: {e}")

    for name, pvh in pvh_results.items():
        plt.plot(pvh["BQML"], pvh["RelativeCumVolume"], label=name)

    plt.xlabel("PET Activity (BQML)")
    plt.ylabel("Relative Volume")
    plt.title("Cumulative PVHs")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    rel_plot_path_PVH = os.path.join("generated_data/PVH", "CPVH_Converted.png")
    plt.savefig(rel_plot_path_PVH, dpi=300, bbox_inches='tight')

    print(f" Relative DVH plot saved to: {rel_plot_path_PVH}")

    try:
        ref_path = os.path.join("generated_data/PVH", "CumulativePVH_AllStructures_RelativeUnits.csv")
        ref_df = pd.read_csv(ref_path, comment='#')
        
        bq_col = next((c for c in ref_df.columns if "bq" in c.lower()), None)
        if not bq_col:
            raise KeyError("No PET activity column found in reference PVH.")

        activity_ref = ref_df[bq_col].values
        diff_output = {bq_col: activity_ref}

        for name, pvh in pvh_results.items():
            clean_name = normalize_structure_name(name)
            rel_col = f"{clean_name}_RelativeCumVolume"
            if rel_col not in ref_df.columns:
                print(f"Structure '{name}' not found in reference PVH. Skipping diff.")
                continue

            interp = np.interp(activity_ref, pvh["BQML"], pvh["RelativeCumVolume"], left=np.nan, right=np.nan)
            ref_values = ref_df[rel_col].values
            difference = interp - ref_values

            diff_output[f"{name}_Diff"] = difference

        diff_df = pd.DataFrame(diff_output)
        diff_path = os.path.join("generated_data/PVH", "PVH_Differences.csv")
        diff_df.to_csv(diff_path, index=False)
        print(f"PVH differences saved to: {diff_path}")

    except Exception as e:
        print(f"Error comparing to reference PVH: {e}")
    

def plot_dvh_and_pvh_differences():
    dvh_diff_path = "generated_data/DVH/DVH_Differences.csv"
    pvh_diff_path = "generated_data/PVH/PVH_Differences.csv"

    dvh_exists = os.path.exists(dvh_diff_path)
    pvh_exists = os.path.exists(pvh_diff_path)

    dvh_df = pd.read_csv(dvh_diff_path) if dvh_exists else None
    pvh_df = pd.read_csv(pvh_diff_path) if pvh_exists else None

    if dvh_df is not None:
        plt.figure()
        plt.clf()
        plt.cla()
        x_col = "GY"

        for column in dvh_df.columns:
            if column != x_col:
                plt.plot(dvh_df[x_col], dvh_df[column], label=column)

        plt.xlabel("Dose (Gy)")
        plt.ylabel("Difference in Relative Volume")
        plt.title("DVH Differences")
        plt.grid(True, alpha=0.3)

        # Deduplicate legend entries (optional but good practice)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        dvh_differences_path = os.path.join("generated_data/DVH", "DVH_Difference_Plot.png")
        plt.savefig(dvh_differences_path, dpi=300, bbox_inches='tight')
        plt.close()

    if pvh_df is not None:
        plt.figure()
        plt.clf()
        plt.cla()
        x_col = next((c for c in pvh_df.columns if "bq" in c.lower()), pvh_df.columns[0])

        for column in pvh_df.columns:
            if column != x_col:
                plt.plot(pvh_df[x_col], pvh_df[column], label=column)

        plt.xlabel("PET Activity (BQML)")
        plt.ylabel("Difference in Relative Volume")
        plt.title("PVH Differences")
        plt.grid(True, alpha=0.3)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        pvh_differences_path = os.path.join("generated_data/PVH", "PVH_Differences_Plot.png")
        plt.savefig(pvh_differences_path, dpi=300, bbox_inches='tight')
        plt.close()