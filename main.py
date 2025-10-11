import sys
import os
from ddvh2cdvh import compare_relative_dvh, compare_relative_pvh, plot_dvh_and_pvh_differences
from helper.load_dicom import read_dicom
from processing.pvh import process_pvh, ask_choice
from processing.dvh import process_dvh
from change_analysis import compare_timepoints_interactive
from userInterface.gui import choose_rtstruct_gui
from helper.structureDICOM import groupDataUID, getValidRS
import tkinter as tk
from tkinter import filedialog, messagebox
import pydicom

# ============================================================================
# NEW: Directory Browser and Patient/Timepoint Discovery
# ============================================================================

def browse_root_directory():
    """
    Opens a dialog to select root directory containing patient data.
    Returns the selected path or None if cancelled.
    """
    root = tk.Tk()
    root.withdraw()
    
    directory = filedialog.askdirectory(
        title="Select Root Directory Containing Patient Data",
        initialdir=os.getcwd()
    )
    
    root.destroy()
    return directory if directory else None


def discover_patients_and_timepoints(root_dir):
    """
    Scans root directory to discover patients and their timepoints.
    
    Returns:
        dict: {
            'Patient_001': {
                'timepoints': {
                    'Timepoint_1': '/path/to/timepoint1',
                    'Timepoint_2': '/path/to/timepoint2'
                },
                'metadata': {...}
            },
            ...
        }
    """
    patients = {}
    
    print("\n" + "="*80)
    print("SCANNING ROOT DIRECTORY FOR PATIENTS AND TIMEPOINTS")
    print("="*80)
    print(f"Root directory: {root_dir}\n")
    
    # Walk through directory structure
    for entry in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, entry)
        
        if not os.path.isdir(patient_path):
            continue
            
        # Check if this directory contains DICOM files or subdirectories
        timepoints = discover_timepoints(patient_path)
        
        if timepoints:
            patients[entry] = {
                'path': patient_path,
                'timepoints': timepoints,
                'metadata': {}
            }
            print(f"✅ Found patient: {entry}")
            for tp_name, tp_path in timepoints.items():
                print(f"   └─ {tp_name}: {tp_path}")
    
    if not patients:
        print("\n❌ No patients with valid DICOM data found!")
        return None
    
    print(f"\n{'='*80}")
    print(f"DISCOVERY COMPLETE: Found {len(patients)} patient(s)")
    print(f"{'='*80}\n")
    
    return patients


def discover_timepoints(patient_dir):
    """
    Discovers timepoints within a patient directory.
    Supports both:
    1. Direct DICOM files in patient folder (single timepoint)
    2. Subdirectories representing different timepoints
    
    Returns:
        dict: {'Timepoint_1': path, 'Timepoint_2': path, ...}
    """
    timepoints = {}
    
    # Check for direct DICOM files (single timepoint scenario)
    has_direct_dicom = has_dicom_files(patient_dir)
    
    if has_direct_dicom:
        # Single timepoint in patient root
        timepoints['Timepoint_1'] = patient_dir
        return timepoints
    
    # Look for timepoint subdirectories
    subdirs = [d for d in os.listdir(patient_dir) 
               if os.path.isdir(os.path.join(patient_dir, d))]
    
    # Sort to ensure consistent ordering (Timepoint_1, Timepoint_2, etc.)
    subdirs.sort()
    
    for subdir in subdirs:
        subdir_path = os.path.join(patient_dir, subdir)
        if has_dicom_files(subdir_path):
            timepoints[subdir] = subdir_path
    
    return timepoints


def has_dicom_files(directory):
    """
    Checks if directory or its subdirectories contain DICOM files.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                # Try to read as DICOM
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                if hasattr(ds, 'Modality'):
                    return True
            except:
                continue
    return False


def analyze_timepoint_data(timepoint_path):
    """
    Analyzes a timepoint directory to identify available modalities.
    
    Returns:
        dict: {
            'has_ct': bool,
            'has_pet': bool,
            'has_rtstruct': bool,
            'has_rtdose': bool,
            'has_reg': bool,
            'ct_count': int,
            'pet_count': int,
            'rtstruct_count': int
        }
    """
    analysis = {
        'has_ct': False,
        'has_pet': False,
        'has_rtstruct': False,
        'has_rtdose': False,
        'has_reg': False,
        'ct_count': 0,
        'pet_count': 0,
        'rtstruct_count': 0
    }
    
    for root, dirs, files in os.walk(timepoint_path):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                modality = getattr(ds, 'Modality', None)
                
                if modality == 'CT':
                    analysis['has_ct'] = True
                    analysis['ct_count'] += 1
                elif modality == 'PT':
                    analysis['has_pet'] = True
                    analysis['pet_count'] += 1
                elif modality == 'RTSTRUCT':
                    analysis['has_rtstruct'] = True
                    analysis['rtstruct_count'] += 1
                elif modality == 'RTDOSE':
                    analysis['has_rtdose'] = True
                elif modality == 'REG':
                    analysis['has_reg'] = True
            except:
                continue
    
    return analysis


def validate_timepoint_connections(patients_data):
    """
    Validates that each timepoint has proper CT/PET/RTSTRUCT connections.
    
    Returns:
        dict: Validation report for each patient/timepoint
    """
    print("\n" + "="*80)
    print("VALIDATING CT/PET/RTSTRUCT CONNECTIONS")
    print("="*80 + "\n")
    
    validation_report = {}
    
    for patient_name, patient_info in patients_data.items():
        print(f"Patient: {patient_name}")
        validation_report[patient_name] = {}
        
        for tp_name, tp_path in patient_info['timepoints'].items():
            analysis = analyze_timepoint_data(tp_path)
            
            # Determine validity
            is_valid = analysis['has_ct'] or analysis['has_pet']
            has_structures = analysis['has_rtstruct']
            
            status = "✅ VALID" if is_valid else "❌ INVALID"
            
            print(f"  {tp_name}: {status}")
            print(f"    CT: {'✅' if analysis['has_ct'] else '❌'} ({analysis['ct_count']} files)")
            print(f"    PET: {'✅' if analysis['has_pet'] else '❌'} ({analysis['pet_count']} files)")
            print(f"    RTSTRUCT: {'✅' if analysis['has_rtstruct'] else '❌'} ({analysis['rtstruct_count']} files)")
            print(f"    RTDOSE: {'✅' if analysis['has_rtdose'] else '❌'}")
            print(f"    REG: {'✅' if analysis['has_reg'] else '❌'}")
            
            validation_report[patient_name][tp_name] = {
                'is_valid': is_valid,
                'analysis': analysis,
                'path': tp_path
            }
        
        print()
    
    print("="*80 + "\n")
    return validation_report


def select_patient_and_timepoints(patients_data, validation_report):
    """
    GUI to select patient and timepoints for analysis.
    
    Returns:
        tuple: (patient_name, timepoint1_path, timepoint2_path or None)
    """
    root = tk.Tk()
    root.title("Select Patient and Timepoints")
    root.geometry("600x400")
    
    selected = {'patient': None, 'tp1': None, 'tp2': None}
    
    # Patient selection
    tk.Label(root, text="Select Patient:", font=('Arial', 12, 'bold')).pack(pady=10)
    
    patient_var = tk.StringVar()
    patient_listbox = tk.Listbox(root, height=5, font=('Arial', 10))
    patient_listbox.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
    
    for patient_name in patients_data.keys():
        patient_listbox.insert(tk.END, patient_name)
    
    # Timepoint selection frame
    tp_frame = tk.Frame(root)
    tp_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
    
    tk.Label(tp_frame, text="Timepoint 1:", font=('Arial', 10)).grid(row=0, column=0, sticky='w')
    tp1_var = tk.StringVar()
    tp1_combo = tk.Listbox(tp_frame, height=3, font=('Arial', 9))
    tp1_combo.grid(row=0, column=1, padx=10, sticky='ew')
    
    tk.Label(tp_frame, text="Timepoint 2 (optional):", font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=10)
    tp2_var = tk.StringVar()
    tp2_combo = tk.Listbox(tp_frame, height=3, font=('Arial', 9))
    tp2_combo.grid(row=1, column=1, padx=10, sticky='ew')
    
    tp_frame.columnconfigure(1, weight=1)
    
    def on_patient_select(event):
        selection = patient_listbox.curselection()
        if not selection:
            return
        
        patient_name = patient_listbox.get(selection[0])
        
        # Update timepoint lists
        tp1_combo.delete(0, tk.END)
        tp2_combo.delete(0, tk.END)
        tp2_combo.insert(tk.END, "None (single timepoint)")
        
        timepoints = patients_data[patient_name]['timepoints']
        for tp_name in timepoints.keys():
            tp1_combo.insert(tk.END, tp_name)
            tp2_combo.insert(tk.END, tp_name)
    
    patient_listbox.bind('<<ListboxSelect>>', on_patient_select)
    
    def on_submit():
        p_sel = patient_listbox.curselection()
        tp1_sel = tp1_combo.curselection()
        
        if not p_sel or not tp1_sel:
            messagebox.showerror("Error", "Please select a patient and at least Timepoint 1")
            return
        
        patient_name = patient_listbox.get(p_sel[0])
        tp1_name = tp1_combo.get(tp1_sel[0])
        
        tp2_sel = tp2_combo.curselection()
        tp2_name = None
        if tp2_sel:
            tp2_text = tp2_combo.get(tp2_sel[0])
            if tp2_text != "None (single timepoint)":
                tp2_name = tp2_text
        
        selected['patient'] = patient_name
        selected['tp1'] = patients_data[patient_name]['timepoints'][tp1_name]
        selected['tp2'] = patients_data[patient_name]['timepoints'][tp2_name] if tp2_name else None
        
        root.quit()
        root.destroy()
    
    tk.Button(root, text="Start Analysis", command=on_submit, 
              bg='green', fg='white', font=('Arial', 11, 'bold')).pack(pady=20)
    
    root.mainloop()
    
    return selected['patient'], selected['tp1'], selected['tp2']


# ============================================================================
# EXISTING FUNCTIONS (kept from original main.py)
# ============================================================================

def load_and_validate_patient(patient_dir, patient_label="Patient"):
    """Load DICOM data and validate RTSTRUCT availability."""
    print(f"\n{'='*80}")
    print(f"LOADING {patient_label.upper()} DATA")
    print(f"{'='*80}")
    
    ct_datasets, pet_datasets, rs_datasets, rd_dataset, reg_dataset = read_dicom(patient_dir)
    
    if not rs_datasets:
        print(f"❌ No RTSTRUCT files found for {patient_label}.")
        return None
    
    return ct_datasets, pet_datasets, rs_datasets, rd_dataset, reg_dataset


def select_rtstruct(rs_datasets, ct_datasets, pet_datasets, patient_label="Patient"):
    """Group datasets and allow user to select an RTSTRUCT."""
    ct_by_uid, pet_by_uid = groupDataUID(ct_datasets, pet_datasets)
    valid_rs_options, valid_rs_options_print = getValidRS(rs_datasets, ct_by_uid, pet_by_uid)
    
    if not valid_rs_options:
        print(f"❌ No valid RTSTRUCTs with matching images for {patient_label}.")
        return None, None, None, None
    
    if len(valid_rs_options) == 1:
        print(f"ℹ️ Only one RTSTRUCT available for {patient_label}, using it.")
        selected = 0
    else:
        selected = choose_rtstruct_gui(valid_rs_options, valid_rs_options_print)
        if selected is None:
            print(f"❌ Invalid RTSTRUCT selection or user cancelled for {patient_label}.")
            return None, None, None, None
    
    rs_dataset, selected_uid = valid_rs_options[selected]
    matched_ct = ct_by_uid.get(selected_uid, [])
    matched_pet = pet_by_uid.get(selected_uid, [])
    
    if not matched_ct and not matched_pet:
        print(f"❌ Selected RTSTRUCT for {patient_label} is not linked to any CT or PET series.")
        return None, None, None, None
    
    print(f"\n✅ Using RTSTRUCT for {patient_label} with UID={selected_uid}")
    print(f"  -> CT slices: {len(matched_ct)}")
    print(f"  -> PET slices: {len(matched_pet)}")
    
    return rs_dataset, matched_ct, matched_pet, selected_uid


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("PET/CT DICOM ANALYSIS TOOL - MULTI-TIMEPOINT SUPPORT")
    print("="*80)
    
    # Step 1: Browse for root directory
    root_dir = browse_root_directory()
    
    if not root_dir:
        print("No directory selected. Exiting.")
        sys.exit()
    
    # Step 2: Discover patients and timepoints
    patients_data = discover_patients_and_timepoints(root_dir)
    
    if not patients_data:
        sys.exit()
    
    # Step 3: Validate connections
    validation_report = validate_timepoint_connections(patients_data)
    
    # Step 4: Select patient and timepoints
    patient_name, tp1_path, tp2_path = select_patient_and_timepoints(
        patients_data, validation_report
    )
    
    if not patient_name or not tp1_path:
        print("No valid selection made. Exiting.")
        sys.exit()
    
    print(f"\n{'='*80}")
    print(f"PROCESSING PATIENT: {patient_name}")
    print(f"{'='*80}")
    print(f"Timepoint 1: {tp1_path}")
    if tp2_path:
        print(f"Timepoint 2: {tp2_path}")
    print(f"{'='*80}\n")
    
    # Step 5: Process Timepoint 1
    patient_data = load_and_validate_patient(tp1_path, "Timepoint 1")
    if patient_data is None:
        sys.exit()
    
    ct_datasets, pet_datasets, rs_datasets, rd_dataset, reg_dataset = patient_data
    
    rs_dataset, matched_ct, matched_pet, selected_uid = select_rtstruct(
        rs_datasets, ct_datasets, pet_datasets, "Timepoint 1"
    )
    
    if rs_dataset is None:
        sys.exit()
    
    # Generate DVH and PVH for Timepoint 1
    process_dvh(rd_dataset, rs_dataset, False)
    curr_pet_volume, curr_masks, curr_structs = process_pvh(
        matched_ct, matched_pet, rd_dataset, reg_dataset, rs_dataset
    )
    
    # Step 6: Process Timepoint 2 (if selected)
    if tp2_path:
        compare_choice = ask_choice("Would you like to compare with Timepoint 2?")
        
        if compare_choice and compare_choice.strip().upper() == 'Y':
            patient2_data = load_and_validate_patient(tp2_path, "Timepoint 2")
            
            if patient2_data is None:
                print("[Compare] Skipping comparison due to data loading error.")
            else:
                ct2, pet2, rs2, rd2, reg2 = patient2_data
                
                rs2_dataset, matched_ct2, matched_pet2, selected_uid2 = select_rtstruct(
                    rs2, ct2, pet2, "Timepoint 2"
                )
                
                if rs2_dataset is None:
                    print("[Compare] Skipping comparison due to RTSTRUCT selection error.")
                else:
                    prev_pet_volume, prev_masks, prev_structs = process_pvh(
                        matched_ct2, matched_pet2, rd2, reg2, rs2_dataset, True
                    )
                    
                    compare_timepoints_interactive(
                        "generated_data",
                        "generated_data2",
                        curr_pet_volume=curr_pet_volume,
                        structure_masks=curr_masks,
                        structures=curr_structs,
                        prev_pet_volume=prev_pet_volume
                    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
