import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons, Button
from matplotlib.patches import Circle
from skimage.measure import label, regionprops
import numpy as np

def choose_rtstruct_gui(options, valid_rs_options_print):
    root = tk.Tk()
    root.withdraw()

    # Combine all printed RTSTRUCT info into one string
    full_text = "\n\n".join(valid_rs_options_print)

    user_input = simpledialog.askstring(
        title="Select RTSTRUCT",
        prompt=f"{full_text}\n\nEnter the index of the RTSTRUCT to use:"
    )

    try:
        index = int(user_input)
        if 0 <= index < len(options):
            return index
        else:
            messagebox.showerror("Invalid Selection", f"Index must be between 0 and {len(options) - 1}")
    except (TypeError, ValueError):
        messagebox.showerror("Invalid Input", "Please enter a valid numeric index.")

    return None

def select_patient_gui(patients_dict):
    """
    GUI to select a patient from discovered patients.
    """
    root = tk.Tk()
    root.title("Select Patient")
    root.geometry("500x400")
    
    selected = {'patient': None}
    
    tk.Label(root, text="Select Patient to Analyze:", font=('Arial', 12, 'bold')).pack(pady=10)
    
    listbox = tk.Listbox(root, height=15, font=('Arial', 10))
    listbox.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
    
    for patient_name, patient_info in patients_dict.items():
        num_timepoints = len(patient_info['timepoints'])
        display_text = f"{patient_name} ({num_timepoints} timepoint{'s' if num_timepoints > 1 else ''})"
        listbox.insert(tk.END, display_text)
    
    def on_submit():
        selection = listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a patient")
            return
        
        patient_names = list(patients_dict.keys())
        selected['patient'] = patient_names[selection[0]]
        root.quit()
        root.destroy()
    
    tk.Button(root, text="Select", command=on_submit,
              bg='green', fg='white', font=('Arial', 11, 'bold')).pack(pady=20)
    
    root.mainloop()
    
    try:
        root.destroy()
    except:
        pass
    
    return selected['patient']

def interactive_change_viewer(curr_pet_volume, prev_pet_volume, structure_masks, structures,
                              init_mode="absolute"):
    """Interactive viewer for changes in PET activity between two 3D stacks with hotspot detection."""

    if curr_pet_volume is None or prev_pet_volume is None:
        print("[Viewer] Need both current and previous PET volumes; skipping.")
        return
    if curr_pet_volume.shape != prev_pet_volume.shape:
        print("[Viewer] Volumes have different shapes; skipping.")
        return
    if not structure_masks:
        print("[Viewer] No structure masks provided; skipping.")
        return

    with np.errstate(divide='ignore', invalid='ignore'):
        delta = curr_pet_volume.astype(np.float32) - prev_pet_volume.astype(np.float32)
        pct = np.where(prev_pet_volume != 0, (delta / prev_pet_volume) * 100.0, 0.0).astype(np.float32)

    mode_names = ["absolute", "percent"]
    mode_idx = 0 if init_mode not in mode_names else mode_names.index(init_mode)
    change_stack = {"absolute": delta, "percent": pct}

    struct_ids = list(structure_masks.keys())
    struct_names = [structures[sid]['name'] for sid in struct_ids]
    selected_structs = [struct_ids[0], struct_ids[1] if len(struct_ids) > 1 else struct_ids[0]]

    z_slices = curr_pet_volume.shape[0]
    slice_idx = z_slices // 2

    diverging_maps = ['coolwarm', 'seismic', 'bwr', 'PiYG', 'PRGn']
    cmap_name = diverging_maps[0]

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
        finite_values = change_slice[np.isfinite(change_slice)]
        if len(finite_values) == 0:
            return [], []
        
        pos_changes = finite_values[finite_values > 0]
        neg_changes = finite_values[finite_values < 0]
        
        hotspots_increase = []
        hotspots_decrease = []
        
        if len(pos_changes) > 0:
            pos_threshold = np.percentile(pos_changes, threshold_percentile)
            pos_mask = (change_slice >= pos_threshold) & np.isfinite(change_slice)
            labeled_pos = label(pos_mask)
            
            for region in regionprops(labeled_pos):
                if region.area >= min_size:
                    y, x = region.centroid
                    intensity = np.mean(change_slice[labeled_pos == region.label])
                    hotspots_increase.append((x, y, intensity, region.area))
        
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

    ax_slice = plt.axes([0.35, 0.25, 0.6, 0.03])
    slice_slider = Slider(ax_slice, 'Slice', 0, z_slices - 1, valinit=slice_idx, valfmt='%0.0f')

    ax_struct = plt.axes([0.025, 0.5, 0.25, 0.4], frameon=False)
    struct_check = CheckButtons(ax_struct, struct_names, [sid in selected_structs for sid in struct_ids])

    ax_cmap = plt.axes([0.025, 0.35, 0.25, 0.12], frameon=False)
    cmap_radio = RadioButtons(ax_cmap, diverging_maps, active=diverging_maps.index(cmap_name))

    ax_mode = plt.axes([0.025, 0.25, 0.25, 0.08], frameon=False)
    mode_radio = RadioButtons(ax_mode, mode_names, active=mode_idx)

    ax_hotspot_btn = plt.axes([0.025, 0.15, 0.12, 0.04])
    hotspot_btn = Button(ax_hotspot_btn, 'Toggle Hotspots')

    ax_threshold = plt.axes([0.15, 0.15, 0.15, 0.03])
    threshold_slider = Slider(ax_threshold, 'Threshold %', 90, 99, valinit=hotspot_threshold_percentile, valfmt='%0.0f')

    ax_min_size = plt.axes([0.025, 0.1, 0.25, 0.03])
    size_slider = Slider(ax_min_size, 'Min Size', 1, 20, valinit=min_hotspot_size, valfmt='%0.0f')

    def draw_overlays(z):
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
        
        for x, y, intensity, area in hotspots_inc:
            circle = Circle((x, y), radius=np.sqrt(area/np.pi), 
                          fill=False, color='orange', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, f'+{intensity:.1f}', ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=8)
        
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
