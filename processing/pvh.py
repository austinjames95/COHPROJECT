from dicompylercore import dicomparser, dvhcalc
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
import os
from matplotlib.path import Path
from helper.binary_masks import create_structure_masks_with_registration
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
import matplotlib.colors as mcolors
from config import elastix_file, elastix_output
from helper.new_mask import add_structure_generation_to_pvh
import tkinter as tk
from tkinter import simpledialog, messagebox

try:
    from helper.resample import Volume
    RESAMPLE_AVAILABLE = True
except ImportError:
    print("Warning: resample module not found - using basic volume creation")
    RESAMPLE_AVAILABLE = False

try:
    from helper.binary_masks import create_structure_masks
    MASKS_AVAILABLE = True
except ImportError:
    print("Warning: binary_masks module not found - using basic mask creation")
    MASKS_AVAILABLE = False

def ask_choice(question_text):
    result = {"choice": None}

    def set_choice(val):
        result["choice"] = val
        root.quit()

    root = tk.Tk()
    root.title("Please Confirm")
    root.geometry("350x120")
    root.eval('tk::PlaceWindow . center')

    tk.Label(root, text=question_text, wraplength=320, justify="center").pack(pady=15)

    frame = tk.Frame(root)
    frame.pack(pady=5)

    tk.Button(frame, text="Yes", width=12, command=lambda: set_choice("Y")).pack(side="left", padx=10)
    tk.Button(frame, text="No", width=12, command=lambda: set_choice("N")).pack(side="right", padx=10)

    root.mainloop()
    root.destroy()

    return result["choice"]

def get_datetime_difference_seconds():
    result = {'diff_seconds': None}

    def on_submit():
        try:
            start_dt = datetime(
                int(start_year.get()), int(start_month.get()), int(start_day.get()),
                int(start_hour.get()), int(start_minute.get())
            )
            end_dt = datetime(
                int(end_year.get()), int(end_month.get()), int(end_day.get()),
                int(end_hour.get()), int(end_minute.get())
            )
            diff = (end_dt - start_dt).total_seconds() / 3600
            result['diff_seconds'] = diff
            result['start'] = start_dt
            result['end'] = end_dt
            root.quit()
        except Exception as e:
            messagebox.showerror("Invalid Input", f"Please check your entries:\n{e}")

    root = tk.Tk()
    root.title("Enter Start and End Date-Time")

    def add_datetime_row(parent, label, row):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=(5, 0))
        month = tk.Entry(parent, width=3)
        day = tk.Entry(parent, width=3)
        year = tk.Entry(parent, width=5)
        hour = tk.Entry(parent, width=3)
        minute = tk.Entry(parent, width=3)
        widgets = [month, day, year, hour, minute]
        for i, widget in enumerate(widgets, start=1):
            widget.grid(row=row, column=i, padx=(0 if i == 1 else 2, 2))
        return widgets

    # Start time
    tk.Label(root, text="Start Date-Time (MM-DD-YYYY HH:MM):").grid(row=0, column=0, columnspan=7, pady=(10, 0))
    start_month, start_day, start_year, start_hour, start_minute = add_datetime_row(root, "Start:", 1)

    # End time
    tk.Label(root, text="End Date-Time (MM-DD-YYYY HH:MM):").grid(row=2, column=0, columnspan=7, pady=(10, 0))
    end_month, end_day, end_year, end_hour, end_minute = add_datetime_row(root, "End:", 3)

    # Submit
    tk.Button(root, text="Submit", command=on_submit).grid(row=4, column=0, columnspan=7, pady=15)

    root.mainloop()
    root.destroy()

    return result['diff_seconds'], result['start'], result['end']

def choose_structure_gui(struct_list):
    """
    Opens a pop-up window to choose a structure from struct_list.
    Returns:
        - None → if the user chooses 0 (skip)
        - A structure name → if a valid index is chosen
        - "ALL" → if user enters a number > len(struct_list)
    """

    root = tk.Tk()
    root.withdraw()

    choices = [f"{i}. {name}" for i, (_, name) in enumerate(struct_list, 1)]
    choice_str = "\n".join(choices)

    user_choice = simpledialog.askstring(
        "Select Reference Structure",
        f"Available Structures:\n0. Skip structure generation\n{choice_str}\n\n"
        f"Enter a number (1-{len(struct_list)}) or a higher number to select ALL structures:"
    )

    try:
        choice_num = int(user_choice)
        if choice_num == 0:
            return None
        elif 1 <= choice_num <= len(struct_list):
            return struct_list[choice_num - 1][1]
        elif choice_num > len(struct_list):
            return "ALL"
        else:
            messagebox.showerror("Invalid Choice", "Please enter a valid number.")
            return choose_structure_gui(struct_list)
    except Exception:
        messagebox.showerror("Invalid Input", "Please enter a number.")
        return choose_structure_gui(struct_list)

def interactive_structure_viewer(pet_volume, structure_masks, structures, secondPatient):
    if not structure_masks:
        print("No structures found to display.")
        return

    struct_ids = list(structure_masks.keys())
    struct_names = [structures[sid]['name'] for sid in struct_ids]

    selected_structs = [struct_ids[0], struct_ids[1] if len(struct_ids) > 1 else struct_ids[0]]

    z_slices = pet_volume.shape[0]
    slice_idx = z_slices // 2

    cmap_name = 'magma'
    pet_clim = (0, 25000)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.35, bottom=0.25)

    img = ax.imshow(
        pet_volume[slice_idx],
        cmap=cmap_name,
        origin='lower',
        vmin=pet_clim[0],
        vmax=pet_clim[1]
    )
    ax.set_title(f"Slice {slice_idx} — Viewing multiple structures")

    # Slice slider
    ax_slice = plt.axes([0.35, 0.1, 0.6, 0.03])
    slice_slider = Slider(ax_slice, 'Slice', 0, z_slices - 1, valinit=slice_idx, valfmt='%0.0f')

    # Structure checkboxes
    ax_struct = plt.axes([0.025, 0.4, 0.25, 0.5], frameon=False)
    struct_check = CheckButtons(ax_struct, struct_names, [sid in selected_structs for sid in struct_ids])

    # Colormap radio
    ax_cmap = plt.axes([0.025, 0.1, 0.25, 0.25], frameon=False)
    colormaps = ['magma', 'plasma', 'inferno', 'viridis', 'cividis']
    cmap_radio = RadioButtons(ax_cmap, colormaps, active=colormaps.index(cmap_name))

    def update_display(z):
        ax.clear()
        ax.imshow(
            pet_volume[z],
            cmap=img.get_cmap(),
            origin='upper',
            vmin=pet_clim[0],
            vmax=pet_clim[1]
        )
        for idx, sid in enumerate(struct_ids):
            if sid in selected_structs:
                color = 'red' if idx % 2 == 0 else 'lime'
                ax.contour(structure_masks[sid][z], colors=color, linewidths=1.5)
        ax.set_title(f"Slice {z} — Viewing: {', '.join(structures[sid]['name'] for sid in selected_structs)}")
        fig.canvas.draw_idle()

    def on_slice_change(val):
        z = int(slice_slider.val)
        update_display(z)

    def on_structure_toggle(label):
        idx = struct_names.index(label)
        sid = struct_ids[idx]
        if sid in selected_structs:
            selected_structs.remove(sid)
        else:
            if len(selected_structs) >= 2:
                selected_structs.pop(0)  # Keep only 2 active
            selected_structs.append(sid)
        update_display(int(slice_slider.val))

    def on_cmap_change(label):
        img.set_cmap(label)
        update_display(int(slice_slider.val))

    slice_slider.on_changed(on_slice_change)
    struct_check.on_clicked(on_structure_toggle)
    cmap_radio.on_clicked(on_cmap_change)

    def on_key(event):
        z = int(slice_slider.val)
        if event.key == 's':
            z = int(slice_slider.val)
            if secondPatient is False:
                filename = f"generated_data/Images/Slice_{z:03d}_Multi.png"
                os.makedirs("generated_data/Images", exist_ok=True)
            else:
                filename = f"generated_data2/Images/Slice_{z:03d}_Multi.png"
                os.makedirs("generated_data2/Images", exist_ok=True)
            fig.savefig(filename, dpi=600, bbox_inches='tight')
            print(f"Saved: {filename}")
        elif event.key == 'right':
            if z < z_slices - 1:
                slice_slider.set_val(z+1)
        elif event.key == 'left':
            if z > 0:
                slice_slider.set_val(z - 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    update_display(slice_idx)
    plt.show()

def get_injection_to_scan_time(pet_dataset):
    try:
        # Try full datetime
        rds = pet_dataset.RadiopharmaceuticalInformationSequence[0]
        inj_time_str = rds.RadiopharmaceuticalStartDateTime  # e.g., '20250721101500.000000'
        inj_time_clean = inj_time_str.split('.')[0]
        inj_time = datetime.strptime(inj_time_clean, '%Y%m%d%H%M%S')

        acq_date = pet_dataset.AcquisitionDate            
        acq_time = pet_dataset.AcquisitionTime.split('.')[0]
        acq_datetime = datetime.strptime(acq_date + acq_time, '%Y%m%d%H%M%S')

        diff_sec = (acq_datetime - inj_time).total_seconds()
        diff_hours = diff_sec / 3600.0

    except Exception as e:
        try:
            # Fallback to time-only (assume same date)
            inj_time_str = rds.RadiopharmaceuticalStartTime
            inj_time_clean = inj_time_str.split('.')[0]

            acq_date = pet_dataset.AcquisitionDate
            acq_time = pet_dataset.AcquisitionTime.split('.')[0]
            
            inj_time = datetime.strptime(acq_date + inj_time_clean, '%Y%m%d%H%M%S')
            acq_datetime = datetime.strptime(acq_date + acq_time, '%Y%m%d%H%M%S')

            diff_sec = (acq_datetime - inj_time).total_seconds()
            diff_hours = diff_sec / 3600.0

        except Exception as inner_e:
            print(f"⚠️  Error parsing injection/acquisition time: {inner_e}")
            print("⚠️  Using fallback decay time: 0 hours")
            manual_entry = ask_choice("Would you like to manually enter the Injection Times?")
            if manual_entry.strip().upper() == 'Y':
                manual_dif, manual_start, manual_end = get_datetime_difference_seconds()
                print("--------------------------------------------------")
                print(f"Injection Time (MANUAL): {manual_start}")
                print(f"Acquisition Date/Time: {manual_end}")
                print(f"Time difference: {manual_dif} hours")
                print("--------------------------------------------------")
                return manual_dif
            else:
                return 1

    # Print result
    print("--------------------------------------------------")
    print(f"Injection Time (DICOM): {inj_time_str} → Parsed: {inj_time}")
    print(f"Acquisition Date/Time: {acq_date} / {acq_time} → Parsed: {acq_datetime}")
    print(f"Time difference: {diff_hours:.3f} hours")
    print("--------------------------------------------------")

    return diff_hours

def create_pet_volume_basic(pet_datasets):
    if not pet_datasets:
        return None, None

    pet_with_position = []
    for ds in pet_datasets:
        try:
            if hasattr(ds, 'ImagePositionPatient'):
                z_pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                z_pos = float(ds.SliceLocation)
            else:
                z_pos = 0.0
            pet_with_position.append((z_pos, ds))
        except:
            pet_with_position.append((0.0, ds))

    pet_with_position.sort(key=lambda x: x[0])
    sorted_pet = [ds for _, ds in pet_with_position]
    first_slice = sorted_pet[0]

    rows = first_slice.Rows
    cols = first_slice.Columns
    num_slices = len(sorted_pet)

    pet_volume = np.zeros((num_slices, rows, cols))

    for i, ds in enumerate(sorted_pet):
        try:
            pixel_array = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            pet_volume[i] = pixel_array
        except Exception as e:
            print(f"Error processing PET slice {i}: {e}")
            pet_volume[i] = np.zeros((rows, cols))

    return pet_volume, sorted_pet

def create_basic_structure_masks(structures, rs_dataset, ct_datasets, volume_shape):
    rt = dicomparser.DicomParser(rs_dataset)
    first_ct = ct_datasets[0]
    origin = np.array(first_ct.ImagePositionPatient)
    spacing = list(map(float, first_ct.PixelSpacing))
    thickness = float(first_ct.SliceThickness)
    spacing.append(thickness)

    z_positions = sorted([float(ds.ImagePositionPatient[2]) for ds in ct_datasets])
    masks = {}

    for sid, struct in structures.items():
        coords = rt.GetStructureCoordinates(sid)
        if not coords:
            continue

        mask = np.zeros(volume_shape, dtype=bool)
        for z_str, contours in coords.items():
            z = float(z_str)
            try:
                k = np.argmin(np.abs(np.array(z_positions) - z))
            except:
                continue

            for contour in contours:
                pts = contour['data']
                if len(pts) < 3:
                    continue

                x = [(pt[0] - origin[0]) / spacing[0] for pt in pts]
                y = [(pt[1] - origin[1]) / spacing[1] for pt in pts]
                poly = Path(np.vstack((x, y)).T)
                grid_x, grid_y = np.meshgrid(np.arange(volume_shape[2]), np.arange(volume_shape[1]))
                points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
                mask2d = poly.contains_points(points).reshape((volume_shape[1], volume_shape[2]))
                mask[k] |= mask2d

        masks[sid] = mask
    return masks

def get_pet_resolution_basic(pet_volume):
    max_pet = np.max(pet_volume)
    min_pet = np.min(pet_volume)
    data_range = max_pet - min_pet
    pet_step = data_range / 1000.0 if data_range > 0 else 1.0
    return pet_step, max_pet

def plot_pvh(pvh_table_absolute, pvh_table_relative, secondPatient):
    num_structures = len([k for k in pvh_table_absolute.keys() if k != 'BQML'])
    colors = plt.cm.tab10(np.linspace(0, 1, num_structures)) if num_structures <= 10 else plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    pet_values = pvh_table_absolute["BQML"]
    plt.figure(figsize=(12, 8))
    for i, (struct_name, volumes) in enumerate(pvh_table_relative.items()):
        if struct_name == "BQML":
            continue
        clean_name = struct_name.replace("_RelativeCumVolume", "").replace("_", " ")
        plt.plot(pvh_table_relative["BQML"], volumes, label=clean_name, linewidth=2, color=colors[i % len(colors)])

    plt.xlabel('PET Activity (Bq/mL)')
    plt.ylabel('Relative Volume (fraction)')
    plt.title('Cumulative PVH - Relative Volumes')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if secondPatient is False:
        os.makedirs("generated_data/PVH", exist_ok=True)
        plt.savefig("generated_data/PVH/PVH_Relative_Plot.png", dpi=300, bbox_inches='tight')
    else:
        os.makedirs("generated_data2/PVH", exist_ok=True)
        plt.savefig("generated_data2/PVH/PVH_Relative_Plot.png", dpi=300, bbox_inches='tight')

    print("Relative PVH plot saved.")

def export_csv(relative_path, pvh_table_relative, custom_bins):
    with open(relative_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(pvh_table_relative.keys()))
        for i in range(len(custom_bins)):
            row = ["{:.6f}".format(pvh_table_relative[h][i]) for h in pvh_table_relative.keys()]
            writer.writerow(row)

def export_csv_suv(relative_path, suv_data):
    with open(relative_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Structure ID", "Structure Name", "Voxel Count", "Volume (mL)", "SUVmean", "SUVmax", "Integral Activity (Bq·mL)"])
        # Write rows
        for entry in suv_data:
            writer.writerow([
                entry["id"],
                entry["name"],
                entry["voxels"],
                f"{entry['volume']:.3f}",
                f"{entry['suv_mean']:.3f}",
                f"{entry['suv_max']:.3f}",
                f"{entry['integral']:.3f}"
            ])

def process_pvh(ct_datasets, pet_datasets, rd_dataset, reg_dataset, rs_dataset, secondPatient=False):
    if not pet_datasets:
        print("\nNo PET data found - skipping PET visualization")
        return

    print(f"\nProcessing {len(pet_datasets)} PET slices...")

    if not rs_dataset:
        print("Warning: No RT Structure Set found - cannot generate PVH")
        return

    # Step 1: Determine RTSTRUCT reference frame (CT or PET)
    rs_references_ct = False
    rs_references_pet = False
    reference_ds = None
    try:
        ct_uids = {ds.SOPInstanceUID for ds in ct_datasets}
        pet_uids = {ds.SOPInstanceUID for ds in pet_datasets}
        referenced_sops = set()

        for roi in getattr(rs_dataset, "ROIContourSequence", []):
            for contour in getattr(roi, "ContourSequence", []):
                for ref in getattr(contour, "ContourImageSequence", []):
                    referenced_sops.add(ref.ReferencedSOPInstanceUID)

        rs_references_ct = any(uid in ct_uids for uid in referenced_sops)
        rs_references_pet = any(uid in pet_uids for uid in referenced_sops)

        if rs_references_pet and not rs_references_ct:
            print("✅ RTSTRUCT is in PET space — no resampling needed.")
            reference_ds = pet_datasets[0]
        elif rs_references_ct and not rs_references_pet:
            print("✅ RTSTRUCT is in CT space — will resample PET to CT grid.")
            reference_ds = ct_datasets[0]
        elif rs_references_ct and rs_references_pet:
            print("⚠️ RTSTRUCT references both CT and PET — defaulting to CT.")
            reference_ds = ct_datasets[0]
        else:
            print("⚠️ Unable to determine RTSTRUCT reference — assuming PET space.")
            rs_references_pet = True
            reference_ds = pet_datasets[0]
    except Exception as e:
        print(f"⚠️ Error determining RTSTRUCT reference modality: {e}")
        rs_references_pet = True
        reference_ds = pet_datasets[0]

    # Step 2: Create PET volume
    if RESAMPLE_AVAILABLE:
        try:
            volume_manager = Volume()
            resampled_pet_volume, sorted_pet = volume_manager.create_pet_volume(pet_datasets)

            if rd_dataset:
                ct_volume, _ = volume_manager.create_ct_volume(ct_datasets)
                resampled_pet_volume = volume_manager.resample_pet_to_ct(
                    pet_datasets, ct_datasets, resampled_pet_volume,
                    manual_transform=None,
                    use_auto_registration=True,
                    ct_volume=ct_volume,
                    use_external_elastix=True,
                    elastix_param_file=elastix_file,
                    elastix_output_dir=elastix_output,
                    reg_dataset=reg_dataset,
                    interpolation='nearest'
                )

        except Exception as e:
            print(f"Error during advanced PET volume creation or resampling: {e}")
            print("Falling back to basic PET volume creation")
            resampled_pet_volume, sorted_pet = create_pet_volume_basic(pet_datasets)
    else:
        print("ℹAdvanced resampling not available — using basic PET volume creation")
        resampled_pet_volume, sorted_pet = create_pet_volume_basic(pet_datasets)

    if resampled_pet_volume is None:
        print("Failed to create PET volume")
        return

    print(f"PET volume shape: {resampled_pet_volume.shape}")

    time_hours = get_injection_to_scan_time(sorted_pet[len(sorted_pet)//2])
    rds = sorted_pet[0].RadiopharmaceuticalInformationSequence[0]

    user_half_life = ask_choice("Zirconium SUVbw Calculation")

    if user_half_life.strip().upper() == 'Y':
        half_life = 78.41
    else:
        half_life = float(rds.RadionuclideHalfLife)
        half_life = half_life / 3600
        print(half_life)
        
    decay_factor = np.exp(-np.log(2) * time_hours / half_life)  # Correct direction: decay back to injection time

    print(f"Time from injection to scan: {time_hours:.2f} hours")
    print(f"Computed decay factor (for correction): {decay_factor:.4f}")

    # Step 4: SUVbw calculation
    weight_kg = float(getattr(sorted_pet[0], 'PatientWeight', 1.0))
    weight_g = weight_kg * 1000
    injected_activity = float(getattr(rds, 'RadionuclideTotalDose', 1.0))
    if injected_activity < 1e6:
        injected_activity *= 1e6  # Convert MBq to Bq

    injected_activity_at_scan = injected_activity * decay_factor

    suv_volume = resampled_pet_volume * weight_g / injected_activity_at_scan

    # Step 5: Get structures
    try:
        dp_struct = dicomparser.DicomParser(rs_dataset)
        structures = dp_struct.GetStructures()
        print(f"Found {len(structures)} structures")
    except Exception as e:
        print(f"Could not get structures: {e}")
        return

    # Step 6: Create structure masks
    try:
        structure_masks = create_structure_masks_with_registration(
            structures, rs_dataset,
            resampled_pet_volume.shape,
            reference_ds=reference_ds,
            volume_instance=volume_manager if RESAMPLE_AVAILABLE else None,
            pet_datasets=pet_datasets,
            ct_datasets=ct_datasets  # Add this parameter
        )
    except Exception as e:
        print(f"Error creating structure masks: {e}")
        return

    # Step 7: Compute PVH
    spacing = list(map(float, reference_ds.PixelSpacing))
    z_positions = sorted([float(ds.ImagePositionPatient[2]) for ds in pet_datasets if hasattr(ds, 'ImagePositionPatient')])
    if len(z_positions) > 1:
        thickness = np.median(np.diff(z_positions))  # More accurate than SliceThickness
    else:
        print("ERROR RESULTING TO 1.0")
        thickness = float(getattr(reference_ds, 'SliceThickness', 1.0))
    voxel_volume_mm3 = spacing[0] * spacing[1] * thickness
    voxel_volume_ml = voxel_volume_mm3 / 1000.0
    print(f"Voxel spacing (mm): {spacing}, thickness (mm): {thickness}, voxel volume: {voxel_volume_ml:.3f} mL")

    pet_step, max_pet = get_pet_resolution_basic(resampled_pet_volume)
    estimated_bins = int(max_pet / pet_step) + 1
    if estimated_bins > 10000:
        print(f"Too many bins ({estimated_bins}), adjusting...")
        pet_step = max_pet / 10000.0

    custom_bins = np.arange(0, max_pet + pet_step, pet_step)
    pvh_table_absolute = {"BQML": custom_bins.tolist()}
    pvh_table_relative = {"BQML": custom_bins.tolist()}

    suv_data = []

    for sid, mask in structure_masks.items():
        if sid not in structures:
            continue

        structure_pet_values = resampled_pet_volume[mask]
        suv_values = suv_volume[mask]

        if structure_pet_values.size == 0:
            print(f"Warning: No voxels found for structure {structures[sid]['name']}")
            continue

        total_voxels = structure_pet_values.size
        total_volume_ml = total_voxels * voxel_volume_ml
        suv_mean = np.mean(suv_values)
        suv_max = np.max(suv_values)
        integral_activity = np.sum(structure_pet_values) * voxel_volume_ml

        cumulative_volumes = []
        relative_volumes = []

        for pet_level in custom_bins:
            voxel_count = np.sum(structure_pet_values >= pet_level)
            volume_ml = voxel_count * voxel_volume_ml
            cumulative_volumes.append(volume_ml)
            relative_volumes.append(volume_ml / total_volume_ml if total_volume_ml > 0 else 0.0)

        name = structures[sid]['name']
        safe_name = name.replace(" ", "_").replace("/", "_")
        pvh_table_absolute[f"{safe_name}_Volume_mL"] = cumulative_volumes
        pvh_table_relative[f"{safe_name}_RelativeCumVolume"] = relative_volumes

        suv_data.append({

            "id": sid,
            "name": name,
            "voxels": total_voxels,
            "volume": total_volume_ml,
            "suv_mean": suv_mean,
            "suv_max": suv_max,
            "integral": integral_activity
        })

    print("\n" + "="*50)
    print("AVAILABLE STRUCTURES FOR PET-BASED GENERATION:")
    print("="*50)

    struct_list = []
    for sid, struct_info in structures.items():
        struct_name = struct_info['name']
        struct_list.append((sid, struct_name))

    reference_structure = choose_structure_gui(struct_list)
    
    if reference_structure == "ALL":
        print(f"\nGenerating PET-based structures for ALL structures ({len(struct_list)} total)")
        
        # Calculate voxel spacing for structure generation
        try:
            spacing = list(map(float, reference_ds.PixelSpacing))
            
            # Get z-spacing from sorted datasets
            if len(pet_datasets) > 1:
                z_positions = []
                for ds in pet_datasets:
                    if hasattr(ds, 'ImagePositionPatient'):
                        z_positions.append(float(ds.ImagePositionPatient[2]))
            
            voxel_spacing_mm = [spacing[0], spacing[1], thickness]
            print(f"Voxel spacing: {voxel_spacing_mm} mm")
            
            # Generate structures for ALL structures
            all_generated_structures = {}
            
            for struct_id, struct_name in struct_list:
                print(f"Generating PET-based structures from: {struct_name}")
                
                try:
                    generated_structures = add_structure_generation_to_pvh(
                        resampled_pet_volume,  
                        structure_masks,      # Existing structure masks
                        structures,          # Structure info dictionary
                        ct_datasets,         # CT datasets for RT struct reference
                        struct_name,         # Current reference structure name
                        voxel_volume_ml,     # Voxel volume in mL
                        voxel_spacing_mm,     # Voxel spacing in mm
                        secondPatient
                    )
                    
                    if generated_structures:
                        # Add with prefix to avoid naming conflicts
                        for gen_name, gen_data in generated_structures.items():
                            prefixed_name = f"{struct_name}_{gen_name}"
                            all_generated_structures[prefixed_name] = gen_data
                        print(f"✅ Generated {len(generated_structures)} structures from {struct_name}")
                    else:
                        print(f"❌ No structures generated from {struct_name}")
                        
                except Exception as struct_gen_error:
                    print(f"❌ Error generating structures from {struct_name}: {struct_gen_error}")
                    continue
            
            if all_generated_structures:
                print(f"✅ Successfully generated {len(all_generated_structures)} new structures total")
                
                # Optionally, add generated structures to the PVH analysis
                include_in_pvh = ask_choice("Do you want to include generated structures in PVH analysis?")
                if include_in_pvh and include_in_pvh.strip().upper() == 'Y':
                    include_in_pvh = True
                else:
                    include_in_pvh = False
                    print("Defaulting to not including in PVH analysis")
                
                include_in_display = ask_choice("Do you want to include the generated structures in the Viewer?")
                if include_in_display and include_in_display.strip().upper() == 'Y':
                    # Add to structures and masks for display
                    for idx, (name, data) in enumerate(all_generated_structures.items()):
                        sid = f"GEN_{idx}"
                        structures[sid] = {"name": name}
                        structure_masks[sid] = data["mask"]
                else:
                    print("Defaulting to non-generated structures")
        
                if include_in_pvh:
                    print("Adding generated structures to PVH analysis...")
                    
                    # Add generated structures to the analysis
                    for struct_name, struct_data in all_generated_structures.items():
                        mask = struct_data['mask']
                        
                        # Get PET values for this structure
                        structure_pet_values = resampled_pet_volume[mask]
                        structure_suv_values = suv_volume[mask]
                        
                        if structure_pet_values.size > 0:
                            # Calculate PVH
                            cumulative_volumes = []
                            relative_volumes = []
                            total_volume_ml = struct_data['stats']['volume_ml']
                            
                            for pet_level in custom_bins:
                                voxel_count = np.sum(structure_pet_values >= pet_level)
                                volume_ml = voxel_count * voxel_volume_ml
                                cumulative_volumes.append(volume_ml)
                                relative_volumes.append(volume_ml / total_volume_ml if total_volume_ml > 0 else 0.0)
                            
                            # Add to PVH tables
                            safe_name = struct_name.replace(" ", "_").replace("/", "_")
                            pvh_table_absolute[f"{safe_name}_Volume_mL"] = cumulative_volumes
                            pvh_table_relative[f"{safe_name}_RelativeCumVolume"] = relative_volumes
                            
                            # Add to SUV data
                            suv_data.append({
                                "id": f"GEN_{len(suv_data)}",  # Generated structure ID
                                "name": struct_name,
                                "voxels": struct_data['stats']['voxel_count'],
                                "volume": struct_data['stats']['volume_ml'],
                                "suv_mean": struct_data['stats']['suv_mean'],
                                "suv_max": struct_data['stats']['suv_max'],
                                "integral": np.sum(structure_pet_values) * voxel_volume_ml
                            })
                    
                    print(f"Added {len(all_generated_structures)} generated structures to PVH analysis")
            else:
                print("❌ No structures were generated from any reference structure")
                
        except Exception as e:
            print(f"Error in ALL structure generation: {e}")
            import traceback
            traceback.print_exc()
    
    elif reference_structure is not None:  # Single structure selected
        print(f"\nGenerating PET-based structures from: {reference_structure}")
        
        # Calculate voxel spacing for structure generation
        try:
            spacing = list(map(float, reference_ds.PixelSpacing))
            
            # Get z-spacing from sorted datasets
            if len(pet_datasets) > 1:
                z_positions = []
                for ds in pet_datasets:
                    if hasattr(ds, 'ImagePositionPatient'):
                        z_positions.append(float(ds.ImagePositionPatient[2]))
            
            voxel_spacing_mm = [spacing[0], spacing[1], thickness]
            print(f"Voxel spacing: {voxel_spacing_mm} mm")
            
            # Generate structures
            try:
                generated_structures = add_structure_generation_to_pvh(
                    resampled_pet_volume,  
                    structure_masks,      # Existing structure masks
                    structures,          # Structure info dictionary
                    ct_datasets,         # CT datasets for RT struct reference
                    reference_structure, # Selected reference structure name
                    voxel_volume_ml,     # Voxel volume in mL
                    voxel_spacing_mm,     # Voxel spacing in mm
                    secondPatient
                )
                
                if generated_structures:
                    print(f"✅ Successfully generated {len(generated_structures)} new structures")
                    
                    # Optionally, add generated structures to the PVH analysis
                    include_in_pvh = ask_choice("Do you want to include generated structures in PVH analysis?")

                    if include_in_pvh and include_in_pvh.strip().upper() == 'Y':
                        include_in_pvh = True
                    else:
                        include_in_pvh = False
                        print("Defaulting to not including in PVH analysis")
                    
                    include_in_display = ask_choice("Do you want to include the generated structures in the Viewer?")

                    if include_in_display and include_in_display.strip().upper() == 'Y':
                        structures.update(generated_structures)
                        for idx, (name, data) in enumerate(generated_structures.items()):
                            sid = f"GEN_{idx}"
                            structures[sid] = {"name": name}
                            structure_masks[sid] = data["mask"]
                    else:
                        print("Defaulting to non-generated structures")
            
                    if include_in_pvh:
                        print("Adding generated structures to PVH analysis...")
                        
                        # Add generated structures to the analysis
                        for struct_name, struct_data in generated_structures.items():
                            mask = struct_data['mask']
                            
                            # Get PET values for this structure
                            structure_pet_values = resampled_pet_volume[mask]
                            structure_suv_values = suv_volume[mask]
                            
                            if structure_pet_values.size > 0:
                                # Calculate PVH
                                cumulative_volumes = []
                                relative_volumes = []
                                total_volume_ml = struct_data['stats']['volume_ml']
                                
                                for pet_level in custom_bins:
                                    voxel_count = np.sum(structure_pet_values >= pet_level)
                                    volume_ml = voxel_count * voxel_volume_ml
                                    cumulative_volumes.append(volume_ml)
                                    relative_volumes.append(volume_ml / total_volume_ml if total_volume_ml > 0 else 0.0)
                                
                                # Add to PVH tables
                                safe_name = struct_name.replace(" ", "_").replace("/", "_")
                                pvh_table_absolute[f"{safe_name}_Volume_mL"] = cumulative_volumes
                                pvh_table_relative[f"{safe_name}_RelativeCumVolume"] = relative_volumes
                                
                                # Add to SUV data
                                suv_data.append({
                                    "id": f"GEN_{len(suv_data)}",  # Generated structure ID
                                    "name": struct_name,
                                    "voxels": struct_data['stats']['voxel_count'],
                                    "volume": struct_data['stats']['volume_ml'],
                                    "suv_mean": struct_data['stats']['suv_mean'],
                                    "suv_max": struct_data['stats']['suv_max'],
                                    "integral": np.sum(structure_pet_values) * voxel_volume_ml
                                })
                        
                        print(f"Added {len(generated_structures)} generated structures to PVH analysis")
                else:
                    print("❌ No structures were generated")
                    
            except Exception as struct_gen_error:
                print(f"❌ Error during structure generation: {struct_gen_error}")
                print("Continuing with original structures only...")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error in structure generation: {e}")
            import traceback
            traceback.print_exc()
    
    if secondPatient is False:
        output_dir = "generated_data/PVH"
        suv_output_dir = "generated_data/SUV_Data"
    else:
        output_dir = "generated_data2/PVH"
        suv_output_dir = "generated_data2/SUV_Data"

    os.makedirs(output_dir, exist_ok=True)
    
    # Export CSV files
    relative_path = os.path.join(output_dir, "CumulativePVH_AllStructures_RelativeUnits.csv")
    export_csv(relative_path, pvh_table_relative, custom_bins)
    
    os.makedirs(suv_output_dir, exist_ok=True)

    suv_relative_path = os.path.join(suv_output_dir, "SUVbw_AllStructures.csv")
    export_csv_suv(suv_relative_path, suv_data)
    
    print(f"\nRelative PVH exported to: {relative_path}")
    print(f"SUV data exported to: {suv_relative_path}")
    print(f"Includes {len(custom_bins)} bins and {len(pvh_table_absolute)-1} structures")
    
    # Generate plots and interactive viewer
    print("\nGenerating PVH plots...")
    plot_pvh(pvh_table_absolute, pvh_table_relative, secondPatient)
    interactive_structure_viewer(resampled_pet_volume, structure_masks, structures, secondPatient)
    return resampled_pet_volume, structure_masks, structures
