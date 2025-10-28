from dicompylercore import dicomparser, dvhcalc
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from processing.plotting import plot_dvh

def export_csv(relative_path_dvh, dvh_table_relative, custom_bins):
    # Ensure output directory exists before writing
    out_dir = os.path.dirname(relative_path_dvh)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

def export_structure_volumes(structure_volumes, output_path):
    """
    Exports structure volumes to CSV file.
    
    Parameters:
    -----------
    structure_volumes : dict
        Dictionary mapping structure names to volumes in mL (cm³)
    output_path : str
        Path to output CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", newline="") as vol_file:
        writer = csv.writer(vol_file)
        writer.writerow(["Structure", "Volume_mL"])
        
        for struct_name, volume in structure_volumes.items():
            writer.writerow([struct_name, f"{volume:.2f}"])
    
    print(f"Structure volumes exported to: {output_path}")

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

def process_dvh(rd_dataset, rs_dataset, secondTimePoint=False):
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
        
        # NEW: Dictionary to store structure volumes
        structure_volumes = {}

        for sid, struct in structures.items():
            try:
                dvh_struct = dvhcalc.get_dvh(dp_struct.ds, dp_dose.ds, sid, calculate_full_volume=True)

                if not dvh_struct:
                    print(f"Warning: No DVH for {struct['name']}")
                    continue

                structure_volume = dvh_struct.volume
                
                # NEW: Store the structure volume
                # Use the same cleaned name format as in the delta file
                clean_struct_name = struct['name']  # Keep original name with spaces
                structure_volumes[clean_struct_name] = structure_volume
                
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

                print(f"  DVH computed for {struct['name']}: {len(absolute_volumes)} dose points, Volume: {structure_volume:.2f} cm³")

            except Exception as e:
                print(f"Error computing DVH for {struct['name']}: {e}")

        if secondTimePoint is False:
            dvh_out_dir = "generated_data/DVH"
        else:
            dvh_out_dir = "generated_data2/DVH"

        relative_path_dvh = os.path.join(dvh_out_dir, "CumulativeDVH_AllStructures_RelativeUnits.csv")

        export_csv(relative_path_dvh, dvh_table_relative, custom_bins)
    
        stats_output_path = os.path.join(dvh_out_dir, "DVH_Structure_Statistics.csv")
        export_dvh_statistics(dvh_table_relative, stats_output_path)
        
        # NEW: Export structure volumes
        volumes_output_path = os.path.join(dvh_out_dir, 'structure_volumes.csv')
        export_structure_volumes(structure_volumes, volumes_output_path)
        
        print(f"\nDVH data exported to: {relative_path_dvh}")
        plot_dvh(dvh_table_absolute, dvh_table_relative, dvh_out_dir)

    except Exception as e:
        print("Error during DVH processing:", e)
