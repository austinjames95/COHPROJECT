
from dicompylercore import dicomparser, dvhcalc
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def plot_dvh(dvh_table_absolute, dvh_table_relative, dvh_out_dir):
    num_structures = len([k for k in dvh_table_absolute.keys() if k != 'GY'])
    if num_structures <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_structures))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    dose_values = dvh_table_absolute["GY"]

    structure_index = 0

    plt.figure(figsize=(12, 8))
    structure_index = 0
    for struct_name, volumes in dvh_table_relative.items():
        if struct_name == "GY":
            continue
        
        clean_name = struct_name.replace("_RelativeCumVolume", "").replace("_", " ")
        plt.plot(dvh_table_relative["GY"], volumes, 
                label=clean_name, 
                linewidth=2,
                color=colors[structure_index % len(colors)])
        structure_index += 1

    plt.xlabel('Dose (Gy)', fontsize=12)
    plt.ylabel('Relative Volume (fraction)', fontsize=12)
    plt.title('Cumulative DVH - Relative Volumes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    rel_plot_path = os.path.join(dvh_out_dir, "DVH_Relative_Plot.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
    print(f" Relative DVH plot saved to: {rel_plot_path}")
    
    print("\nDVH Summary Statistics:")
    print("="*60)
    
    for struct_name, volumes in dvh_table_absolute.items():
        if struct_name == "GY":
            continue
            
        clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
        max_volume = volumes[0] 

        rel_struct_name = struct_name.replace("_Volume_cm3", "")
        if rel_struct_name in dvh_table_relative:
            rel_volumes = dvh_table_relative[rel_struct_name]
            
            dose_95_vol = None
            dose_50_vol = None
            dose_5_vol = None
            
            for j, rel_vol in enumerate(rel_volumes):
                if dose_95_vol is None and rel_vol <= 0.95:
                    dose_95_vol = dose_values[j]
                if dose_50_vol is None and rel_vol <= 0.50:
                    dose_50_vol = dose_values[j]
                if dose_5_vol is None and rel_vol <= 0.05:
                    dose_5_vol = dose_values[j]
        
        print(f"\n{clean_name}:")
        print(f"  Total Volume: {max_volume:.2f} cm³")
        if 'dose_95_vol' in locals() and dose_95_vol is not None:
            print(f"  D95 (dose to 95% volume): {dose_95_vol:.2f} Gy")
        if 'dose_50_vol' in locals() and dose_50_vol is not None:
            print(f"  D50 (dose to 50% volume): {dose_50_vol:.2f} Gy")
        if 'dose_5_vol' in locals() and dose_5_vol is not None:
            print(f"  D5 (dose to 5% volume): {dose_5_vol:.2f} Gy")
    
    print("\n" + "="*60)
    print("\nProcessing completed successfully!")

def export_csv(relative_path_dvh, dvh_table_relative, custom_bins):
    with open(relative_path_dvh, "w", newline="") as f:
        writer = csv.writer(f)
        headers = list(dvh_table_relative.keys())
        writer.writerow(headers)
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "GY":
                    row.append("{:.6f}".format(dvh_table_relative[h][i]))
                else:
                    row.append("{:.6f}".format(dvh_table_relative[h][i]))
            writer.writerow(row)

def export_dvh_statistics(dvh_table_relative, output_path):
    """
    Exports mean dose (Gy) and max dose (Gy) per structure using the cumulative DVH.
    - Mean dose = area under the cumulative DVH curve (trapz over dose).
    - Max dose = highest dose bin where cumulative relative volume > 0 (with epsilon).
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dose_values = np.asarray(dvh_table_relative["GY"], dtype=float)
    EPS = 1e-8  # numerical tolerance

    with open(output_path, "w", newline="") as stats_file:
        writer = csv.writer(stats_file)
        writer.writerow(["Structure", "Mean Dose (Gy)", "Max Dose (Gy)"])

        for struct_key, rel_volumes in dvh_table_relative.items():
            if struct_key == "GY":
                continue

            # Clean display name
            clean_name = struct_key.replace("_RelativeCumVolume", "").replace("_", " ")

            # Ensure numpy arrays
            F = np.asarray(rel_volumes, dtype=float)

            F = np.minimum.accumulate(F)

            F = np.clip(F, 0.0, 1.0)

            mean_dose = np.trapz(F, dose_values)  # Gy

            # --- Max dose: last dose with non-zero cumulative volume ---
            pos = np.where(F > EPS)[0]
            if pos.size > 0:
                max_dose = float(dose_values[pos[-1]])
            else:
                max_dose = 0.0

            writer.writerow([clean_name, f"{mean_dose:.2f}", f"{max_dose:.2f}"])

    print(f"\nStructure dose statistics exported to: {output_path}")

def process_dvh(rd_dataset, rs_dataset, secondPatient=False):
    if not rs_dataset:
        print("RTSTRUCT dataset is required for DVH processing.")
        return

    print("\nParsing DICOM datasets with dicompylercore:")

    try:
        dp_struct = dicomparser.DicomParser(rs_dataset)
        structures = dp_struct.GetStructures()
    except Exception as e:
        print("Failed to parse RTSTRUCT or extract structures:", e)
        return

    if not rd_dataset:
        print("Warning: RTDOSE dataset not provided. Skipping DVH computation.")
        print("\nAvailable Structures:")
        for sid, struct in structures.items():
            print(f"  [{sid}] {struct['name']}")
        return

    try:
        dp_dose = dicomparser.DicomParser(rd_dataset)
    except Exception as e:
        print("Failed to parse RTDOSE:", e)
        return

    # Proceed with DVH computation
    try:
        dose_array = rd_dataset.pixel_array * rd_dataset.DoseGridScaling
        unique_doses = np.unique(dose_array.flatten())
        dose_diffs = np.diff(np.sort(unique_doses))
        min_step = np.min(dose_diffs[dose_diffs > 0])
        dose_step, max_dose = min_step, np.max(unique_doses) 
    
        custom_bins = np.arange(0, max_dose + dose_step, dose_step)

        dose_units = getattr(rd_dataset, 'DoseUnits', 'GY')
        print(f"\nDose units from DICOM: {dose_units}")

        dvh_table_absolute = {"GY": [float(d) for d in custom_bins.tolist()]}
        dvh_table_relative = {"GY": [float(d) for d in custom_bins.tolist()]}

        for sid, struct in structures.items():
            try:
                dvh_struct = dvhcalc.get_dvh(dp_struct.ds, dp_dose.ds, sid, calculate_full_volume=True)

                if not dvh_struct:
                    print(f"Warning: No DVH for {struct['name']}")
                    continue

                structure_volume = dvh_struct.volume
                absolute_volumes = []
                for dose in custom_bins:
                    vol = dvh_struct.volume_constraint(dose, dose_units="Gy")
                    volume_value = float(vol.value) if hasattr(vol, 'value') else float(vol)
                    absolute_volumes.append(volume_value)

                relative_volumes = [vol / structure_volume if structure_volume > 0 else 0.0 for vol in absolute_volumes]

                safe_name = struct['name'].replace(" ", "_").replace("/", "_").replace("-", "_")
                abs_column_name = f"{safe_name}_Volume_cm3"
                rel_column_name = f"{safe_name}_RelativeCumVolume"

                dvh_table_absolute[abs_column_name] = absolute_volumes
                dvh_table_relative[rel_column_name] = relative_volumes

                print(f"  DVH computed for {struct['name']}: {len(absolute_volumes)} dose points")

            except Exception as e:
                print(f"Error computing DVH for {struct['name']}: {e}")

        if secondPatient is False:
            dvh_out_dir = "generated_data/DVH"
        else:
            dvh_out_dir = "generated_data2/DVH"

        relative_path_dvh = os.path.join(dvh_out_dir, "CumulativeDVH_AllStructures_RelativeUnits.csv")

        export_csv(relative_path_dvh, dvh_table_relative, custom_bins)
    
        stats_output_path = os.path.join(dvh_out_dir, "DVH_Structure_Statistics.csv")
        export_dvh_statistics(dvh_table_relative, stats_output_path)
        
        print(f"\nDVH data exported to: {relative_path_dvh}")
        plot_dvh(dvh_table_absolute, dvh_table_relative, dvh_out_dir)

    except Exception as e:
        print("Error during DVH processing:", e)
