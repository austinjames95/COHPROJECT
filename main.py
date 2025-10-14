import os
import sys

# Add lib folder to path
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from ddvh2cdvh import compare_relative_dvh, compare_relative_pvh, plot_dvh_and_pvh_differences
from helper.load_dicom import read_dicom, browse_root_directory, has_dicom_files
from processing.pvh import process_pvh, ask_choice
from processing.dvh import process_dvh
from change_analysis import compare_timepoints_interactive
from userInterface.gui import choose_rtstruct_gui, select_patient_gui
from helper.structureDICOM import groupDataUID, getValidRS
from config import given_dvh_dir, given_pvh_dir, patientOne, patientTwo, compare_data_dir
import pydicom


def analyze_folder_content(folder_path):
    """
    Analyzes a folder to identify what DICOM modalities it contains.
    Returns dict with has_ct, has_pet, has_rtstruct, etc.
    """
    analysis = {
        'has_ct': False,
        'has_pet': False,
        'has_rtstruct': False,
        'has_rtdose': False,
        'has_reg': False,
        'ct_count': 0,
        'pet_count': 0,
        'rtstruct_count': 0,
        'path': folder_path
    }
    
    for root, dirs, files in os.walk(folder_path):
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


def discover_timepoints_in_patient(patient_dir):
    """
    Discovers timepoints within a patient directory.
    
    Strategy:
    1. Each subfolder is examined independently
    2. Folders with CT are considered primary timepoints
    3. PET-only folders are also timepoints (will use reference CT)
    4. Returns dict mapping each folder to its content
    
    Returns: {
        'ct1': {analysis}, 
        'zr70nzr1': {analysis},
        ...
    }
    """
    if not os.path.exists(patient_dir):
        print(f"❌ Patient directory not found: {patient_dir}")
        return {}
    
    timepoint_folders = {}
    
    # Check if patient_dir itself contains DICOM files (single folder case)
    if has_dicom_files(patient_dir, max_depth=0):
        analysis = analyze_folder_content(patient_dir)
        timepoint_folders['ROOT'] = analysis
        return timepoint_folders
    
    # Scan all subdirectories
    try:
        for item in os.listdir(patient_dir):
            item_path = os.path.join(patient_dir, item)
            if os.path.isdir(item_path) and has_dicom_files(item_path):
                analysis = analyze_folder_content(item_path)
                # Only include folders that have CT or PET (actual imaging data)
                if analysis['has_ct'] or analysis['has_pet']:
                    timepoint_folders[item] = analysis
        
    except Exception as e:
        print(f"Error scanning patient directory {patient_dir}: {e}")
        import traceback
        traceback.print_exc()
    
    return timepoint_folders


def identify_timepoint_structure(timepoint_folders, patient_dir):
    """
    Identifies the timepoint structure and determines if CT needs to be shared.
    
    Returns: [
        {
            'name': 'Timepoint_1',
            'folders': ['ct1'],
            'has_ct': True,
            'has_pet': True,
            'reference_ct_folder': None  # Uses own CT
        },
        {
            'name': 'Timepoint_2', 
            'folders': ['zr70nzr1'],
            'has_ct': False,
            'has_pet': True,
            'reference_ct_folder': 'ct1'  # Uses CT from ct1
        }
    ]
    """
    timepoints = []
    
    # Sort folders alphabetically for consistent ordering
    sorted_folders = sorted(timepoint_folders.keys())
    
    # Find the primary CT folder (first folder with CT)
    primary_ct_folder = None
    for folder_name in sorted_folders:
        if timepoint_folders[folder_name]['has_ct']:
            primary_ct_folder = folder_name
            break
    
    # Create timepoint entries
    for idx, folder_name in enumerate(sorted_folders, 1):
        folder_info = timepoint_folders[folder_name]
        
        timepoint = {
            'name': f'Timepoint_{idx}',
            'folders': [folder_name],
            'has_ct': folder_info['has_ct'],
            'has_pet': folder_info['has_pet'],
            'has_rtstruct': folder_info['has_rtstruct'],
            'has_rtdose': folder_info['has_rtdose'],
            'has_reg': folder_info['has_reg'],
            'ct_count': folder_info['ct_count'],
            'pet_count': folder_info['pet_count'],
            'reference_ct_folder': None
        }
        
        # If this timepoint has PET but no CT, reference the primary CT
        if folder_info['has_pet'] and not folder_info['has_ct'] and primary_ct_folder:
            timepoint['reference_ct_folder'] = primary_ct_folder
            timepoint['uses_shared_ct'] = True
        else:
            timepoint['uses_shared_ct'] = False
        
        timepoints.append(timepoint)
    
    return timepoints


def print_timepoint_summary(patient_label, timepoints, patient_dir):
    """
    Prints a summary of discovered timepoints.
    """
    print(f"\n{'='*80}")
    print(f"{patient_label.upper()} TIMEPOINT DISCOVERY")
    print(f"{'='*80}")
    
    if not timepoints:
        print("❌ No valid timepoints found")
        return False
    
    print(f"✅ Found {len(timepoints)} timepoint(s)\n")
    
    for tp in timepoints:
        status = "✅ VALID" if (tp['has_ct'] or tp['has_pet']) else "⚠️  INCOMPLETE"
        
        print(f"{tp['name']}: {status}")
        print(f"  Folder(s): {', '.join(tp['folders'])}")
        print(f"  CT: {'✅' if tp['has_ct'] else '❌'} ({tp['ct_count']} files)")
        print(f"  PET: {'✅' if tp['has_pet'] else '❌'} ({tp['pet_count']} files)")
        print(f"  RTSTRUCT: {'✅' if tp['has_rtstruct'] else '❌'}")
        print(f"  RTDOSE: {'✅' if tp['has_rtdose'] else '❌'}")
        print(f"  REG: {'✅' if tp['has_reg'] else '❌'}")
        
        if tp.get('uses_shared_ct'):
            print(f"  ⚠️  Uses CT from: {tp['reference_ct_folder']}")
        
        print()
    
    return True


def load_timepoint_data(patient_dir, timepoint_info, timepoint_label):
    """
    Loads DICOM data for a timepoint.
    
    If timepoint has no CT, loads CT from reference folder.
    """
    print(f"\n{'='*80}")
    print(f"LOADING {timepoint_label.upper()}")
    print(f"{'='*80}")
    
    all_ct_datasets = []
    all_pet_datasets = []
    all_rs_datasets = []
    rd_dataset = None
    reg_dataset = None
    
    # Load data from timepoint's own folder(s)
    for folder in timepoint_info['folders']:
        folder_path = os.path.join(patient_dir, folder)
        print(f"\nScanning: {folder_path}")
        
        ct, pet, rs, rd, reg = read_dicom(folder_path)
        
        all_ct_datasets.extend(ct)
        all_pet_datasets.extend(pet)
        all_rs_datasets.extend(rs)
        
        if rd is not None:
            rd_dataset = rd
        if reg is not None:
            reg_dataset = reg
    
    # If no CT found but reference CT folder specified, load CT from there
    if not all_ct_datasets and timepoint_info.get('reference_ct_folder'):
        ref_folder = timepoint_info['reference_ct_folder']
        ref_path = os.path.join(patient_dir, ref_folder)
        
        print(f"\n⚠️  No CT in current timepoint. Loading CT from reference: {ref_folder}")
        
        ct_ref, _, rs_ref, rd_ref, _ = read_dicom(ref_path)
        all_ct_datasets.extend(ct_ref)
        
        # Also get RTSTRUCT from reference if current timepoint doesn't have it
        if not all_rs_datasets:
            all_rs_datasets.extend(rs_ref)
            print(f"   Also using RTSTRUCT from {ref_folder}")
        
        # Get RTDOSE from reference if current doesn't have it
        if rd_dataset is None and rd_ref is not None:
            rd_dataset = rd_ref
            print(f"   Also using RTDOSE from {ref_folder}")
    
    if not all_rs_datasets:
        print(f"❌ No RTSTRUCT files found for {timepoint_label}.")
        return None
    
    print(f"\n✅ Combined data for {timepoint_label}:")
    print(f"   CT slices: {len(all_ct_datasets)}")
    print(f"   PET slices: {len(all_pet_datasets)}")
    print(f"   RTSTRUCT files: {len(all_rs_datasets)}")
    print(f"   RTDOSE: {'✅' if rd_dataset else '❌'}")
    print(f"   REG: {'✅' if reg_dataset else '❌'}")
    
    return all_ct_datasets, all_pet_datasets, all_rs_datasets, rd_dataset, reg_dataset


def select_rtstruct(rs_datasets, ct_datasets, pet_datasets, patient_label="Patient"):
    """Group datasets and allow user to select an RTSTRUCT."""
    ct_by_uid, pet_by_uid = groupDataUID(ct_datasets, pet_datasets)
    valid_rs_options, valid_rs_options_print = getValidRS(rs_datasets, ct_by_uid, pet_by_uid)
    
    if not valid_rs_options:
        print(f"❌ No valid RTSTRUCTs with matching images for {patient_label}.")
        return None, None, None, None
    
    if len(valid_rs_options) == 1:
        print(f"ℹOnly one RTSTRUCT available for {patient_label}, using it.")
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


def main():
    print("\n" + "="*80)
    print("PET/CT DICOM ANALYSIS TOOL - MULTI-TIMEPOINT SUPPORT")
    print("="*80)
    
    # Ask user if they want to use automated discovery or config file
    use_automated = ask_choice("Use automated patient discovery? (No = use config.py)")
    
    if use_automated and use_automated.strip().upper() == 'Y':
        # ===== AUTOMATED DISCOVERY MODE =====
        root_dir = browse_root_directory()
        
        if not root_dir:
            print("No directory selected. Falling back to config.py...")
            use_automated = False
        else:
            print(f"\n{'='*80}")
            print("SCANNING ROOT DIRECTORY FOR PATIENTS")
            print(f"{'='*80}")
            print(f"Root: {root_dir}\n")
            
            patients_found = {}
            for entry in os.listdir(root_dir):
                patient_path = os.path.join(root_dir, entry)
                if not os.path.isdir(patient_path):
                    continue
                
                # Discover timepoint folders
                timepoint_folders = discover_timepoints_in_patient(patient_path)
                if timepoint_folders:
                    # Identify structure
                    timepoints = identify_timepoint_structure(timepoint_folders, patient_path)
                    patients_found[entry] = {
                        'path': patient_path,
                        'timepoints': timepoints
                    }
                    print(f"✅ Found patient: {entry} ({len(timepoints)} timepoint(s))")
            
            if not patients_found:
                print("❌ No patients with valid DICOM data found!")
                print("Falling back to config.py...")
                use_automated = False
            else:
                print(f"\n✅ Found {len(patients_found)} patient(s) total")
                
                # Show patient selection GUI
                selected_patient = select_patient_gui(patients_found)
                if not selected_patient:
                    print("No patient selected. Exiting.")
                    sys.exit()
                
                patient_name = selected_patient
                patient_dir = patients_found[patient_name]['path']
                patient_timepoints = patients_found[patient_name]['timepoints']
                
                print_timepoint_summary(patient_name, patient_timepoints, patient_dir)
    
    if not use_automated or use_automated.strip().upper() != 'Y':
        # ===== CONFIG FILE MODE =====
        print("\nUsing config.py for patient directories...")
        
        # Discover timepoints for Patient 1
        timepoint_folders = discover_timepoints_in_patient(patientOne)
        patient_timepoints = identify_timepoint_structure(timepoint_folders, patientOne)
        
        print_timepoint_summary("Patient 1", patient_timepoints, patientOne)
        
        if not patient_timepoints:
            print("\n❌ No valid timepoints found for Patient 1. Exiting.")
            sys.exit()
        
        patient_dir = patientOne
        patient_name = "Patient 1"
    
    # ===== PROCESS TIMEPOINT 1 =====
    print(f"\n{'='*80}")
    print(f"PROCESSING {patient_name.upper()} - TIMEPOINT 1")
    print(f"{'='*80}")
    
    if len(patient_timepoints) == 0:
        print("❌ No timepoints found")
        sys.exit()
    
    tp1_info = patient_timepoints[0]
    
    patient_data = load_timepoint_data(patient_dir, tp1_info, f"{patient_name} - Timepoint 1")
    
    if patient_data is None:
        sys.exit()
    
    ct_datasets, pet_datasets, rs_datasets, rd_dataset, reg_dataset = patient_data
    
    rs_dataset, matched_ct, matched_pet, selected_uid = select_rtstruct(
        rs_datasets, ct_datasets, pet_datasets, f"{patient_name} - Timepoint 1"
    )
    
    if rs_dataset is None:
        sys.exit()
    
    # Generate DVH and PVH
    process_dvh(rd_dataset, rs_dataset, False)
    curr_pet_volume, curr_masks, curr_structs = process_pvh(
        matched_ct, matched_pet, rd_dataset, reg_dataset, rs_dataset
    )
    
    # Check for pre-existing comparison data
    if given_dvh_dir is not None or given_pvh_dir is not None:  
        if given_dvh_dir is not None:
            compare_relative_dvh()
        if given_pvh_dir is not None:
            compare_relative_pvh()
        plot_dvh_and_pvh_differences()
    
    # ===== PROCESS TIMEPOINT 2 (if exists) =====
    if len(patient_timepoints) > 1:
        print(f"\n{'='*80}")
        print(f"MULTIPLE TIMEPOINTS DETECTED")
        print(f"{'='*80}")
        print(f"Found {len(patient_timepoints)} timepoints for {patient_name}")
        
        compare_choice = ask_choice("Would you like to compare Timepoint 1 with Timepoint 2?")
        
        if compare_choice and compare_choice.strip().upper() == 'Y':
            tp2_info = patient_timepoints[1]
            
            print(f"\n{'='*80}")
            print(f"PROCESSING {patient_name.upper()} - TIMEPOINT 2")
            print(f"{'='*80}")
            
            patient2_data = load_timepoint_data(patient_dir, tp2_info, f"{patient_name} - Timepoint 2")
            
            if patient2_data is None:
                print("[Compare] Skipping timepoint comparison due to data loading error.")
                sys.exit()
            else:
                ct2, pet2, rs2, rd2, reg2 = patient2_data # ct_datasets, pet_datasets, rs_datasets, rd_dataset, reg_dataset = patient_data
                
                rs2_dataset, matched_ct2, matched_pet2, selected_uid2 = select_rtstruct(
                    rs2, ct2, pet2, f"{patient_name} - Timepoint 2"
                )
                
                if rs2_dataset is None:
                    print("[Compare] Skipping timepoint comparison due to RTSTRUCT selection error.")
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
