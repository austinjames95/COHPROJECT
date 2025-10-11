import os
import sys
# Add lib folder to path
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from ddvh2cdvh import compare_relative_dvh, compare_relative_pvh, plot_dvh_and_pvh_differences
from helper.load_dicom import read_dicom
from processing.pvh import process_pvh, ask_choice
from processing.dvh import process_dvh
from change_analysis import compare_timepoints_interactive
from userInterface.gui import choose_rtstruct_gui
from helper.structureDICOM import groupDataUID, getValidRS
from config import given_dvh_dir, given_pvh_dir, patientOne, patientTwo, compare_data_dir

# Load and validate patient data 
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
    # Group CT and PET datasets by FrameOfReferenceUID
    ct_by_uid, pet_by_uid = groupDataUID(ct_datasets, pet_datasets)
    
    # Group and show RTSTRUCTs and their linked image sets
    valid_rs_options, valid_rs_options_print = getValidRS(rs_datasets, ct_by_uid, pet_by_uid)
    
    if not valid_rs_options:
        print(f"❌ No valid RTSTRUCTs with matching images for {patient_label}.")
        return None, None, None, None
    
    # Choose which RTSTRUCT file to use
    if len(valid_rs_options) == 1:
        print(f"ℹ️ Only one RTSTRUCT available for {patient_label}, using it.")
        selected = 0
    else:
        selected = choose_rtstruct_gui(valid_rs_options, valid_rs_options_print)
        if selected is None:
            print(f"❌ Invalid RTSTRUCT selection or user cancelled for {patient_label}.")
            return None, None, None, None
    
    # Extract selected data
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

patient_data = load_and_validate_patient(patientOne, "Patient 1")
if patient_data is None:
    sys.exit()

ct_datasets, pet_datasets, rs_datasets, rd_dataset, reg_dataset = patient_data

rs_dataset, matched_ct, matched_pet, selected_uid = select_rtstruct(
    rs_datasets, ct_datasets, pet_datasets, "Patient 1"
)

if rs_dataset is None:
    sys.exit()

# Generate DVH and PVH data for first patient

process_dvh(rd_dataset, rs_dataset, False)
curr_pet_volume, curr_masks, curr_structs = process_pvh(matched_ct, matched_pet, rd_dataset, reg_dataset, rs_dataset)

# check to see if you are comparing to pre-existing dvh / pvh data

if given_dvh_dir is not None or given_pvh_dir is not None:  
    if given_dvh_dir is not None:
        compare_relative_dvh()
    if given_pvh_dir is not None:
        compare_relative_pvh()
    
    plot_dvh_and_pvh_differences()

if patientTwo is None:
    print("\nNo second patient configured. Analysis complete.")
    sys.exit()
    
compare_timepoints = ask_choice("Would you like to compare this to the other patient")
if compare_timepoints.strip().upper() != 'Y':
    print("[Compare] User chose not to compare. Analysis complete.")
    sys.exit()

# Load second patient
patient2_data = load_and_validate_patient(patientTwo, "Patient 2")
if patient2_data is None:
    print("[Compare] Skipping comparison due to data loading error.")
    sys.exit()

ct2, pet2, rs2, rd2, reg2 = patient2_data

rs2_dataset, matched_ct2, matched_pet2, selected_uid2 = select_rtstruct(
    rs2, ct2, pet2, "Patient 2"
)
if rs2_dataset is None:
    print("[Compare] Skipping comparison due to RTSTRUCT selection error.")
    sys.exit()

# Process second patient data

prev_pet_volume, prev_masks, prev_structs = process_pvh(matched_ct2, matched_pet2, rd2, reg2, rs2_dataset, True)

compare_timepoints_interactive(
    "generated_data",
    "generated_data2",
    curr_pet_volume=curr_pet_volume,
    structure_masks=curr_masks,
    structures=curr_structs,
    prev_pet_volume=prev_pet_volume
)
