from ddvh2cdvh import compare_relative_dvh, compare_relative_pvh, plot_dvh_and_pvh_differences
from helper.load_dicom import read_dicom
from processing.pvh import process_pvh, ask_choice
from processing.dvh import process_dvh
from change_analysis import compare_timepoints_interactive
from dicompylercore import dicomparser
import time
from config import given_dvh_dir, given_pvh_dir, patientOne, patientTwo, compare_data_dir
import tkinter as tk
from tkinter import simpledialog, messagebox

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

ct_datasets, pet_datasets, rs_datasets, rd_dataset, reg_dataset = read_dicom(patientOne)

if not rs_datasets:
    print("❌ No RTSTRUCT files found.")
    exit()

# Group CT and PET datasets by FrameOfReferenceUID
ct_by_uid = {}
pet_by_uid = {}
for ds in ct_datasets:
    uid = getattr(ds, 'FrameOfReferenceUID', None)
    if uid:
        ct_by_uid.setdefault(uid, []).append(ds)

for ds in pet_datasets:
    uid = getattr(ds, 'FrameOfReferenceUID', None)
    if uid:
        pet_by_uid.setdefault(uid, []).append(ds)

# Show RTSTRUCTs and their linked image sets
valid_rs_options = []
valid_rs_options_print = []
print("\nMultiple RTSTRUCT files found:")
for i, rs in enumerate(rs_datasets):
    uid = getattr(rs, "FrameOfReferenceUID", "UNKNOWN_UID")
    patient_name = getattr(rs, "PatientName", "Unknown")
    patient_id = getattr(rs, "PatientID", "Unknown")
    study_desc = getattr(rs, "StudyDescription", "No Description")
    modality = getattr(rs, "Modality", "RTSTRUCT")

    try:
        rt = dicomparser.DicomParser(rs)
        structures = rt.GetStructures()
        struct_names = ", ".join([s["name"] for s in structures.values()])
    except:
        struct_names = "N/A"

    ct_match = "✅" if uid in ct_by_uid else "❌"
    pet_match = "✅" if uid in pet_by_uid else "❌"

    label = f"[{len(valid_rs_options)}] {modality} for {patient_name} (ID: {patient_id})"
    label += f"\n ├─ Study: {study_desc}\n ├─ UID: {uid}\n ├─ Linked CT: {ct_match}, PET: {pet_match}"
    label += f"\n └─ Structures: \n     {struct_names}\n"

    valid_rs_options_print.append(label)
    valid_rs_options.append((rs, uid))

if not valid_rs_options:
    print("❌ No valid RTSTRUCTs with matching images.")
    exit()

if len(valid_rs_options) == 1:
    print("⚠️ Only one RTSTRUCT avaliable, defaulting to only possible option")
    selected = 0
    print(patient_id)
else:
    selected = choose_rtstruct_gui(valid_rs_options, valid_rs_options_print)
    if selected is None:
        print("❌ Invalid RTSTRUCT selection or user cancelled.")
        exit()

rs_dataset, selected_uid = valid_rs_options[selected]

matched_ct = ct_by_uid.get(selected_uid, [])
matched_pet = pet_by_uid.get(selected_uid, [])

if not matched_ct and not matched_pet:
    print("❌ Selected RTSTRUCT is not linked to any CT or PET series.")
    exit()

print(f"\n✅ Using RTSTRUCT with UID={selected_uid}")
print(f"  -> CT slices: {len(matched_ct)}")
print(f"  -> PET slices: {len(matched_pet)}")

process_dvh(rd_dataset, rs_dataset, False)
curr_pet_volume, curr_masks, curr_structs = process_pvh(matched_ct, matched_pet, rd_dataset, reg_dataset, rs_dataset)
if given_dvh_dir is not None:
    compare_relative_dvh()
if given_pvh_dir is not None:
    compare_relative_pvh()

time.sleep(5)
if given_pvh_dir is not None or given_dvh_dir is not None:
    plot_dvh_and_pvh_differences()

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)

if patientTwo is not None:
    compare_timepoints = ask_choice("Would you like to compare this to the other patient")
    if compare_timepoints.strip().upper() == 'Y':

        prev_pet_volume = None
        prev_masks = None
        prev_structs = None

        # If a prior analysis exists on disk, we still try to ALSO get a live prev volume
        try:
            # Read patientTwo (or however your config defines the 'previous' dataset)
            ct2, pet2, rs2_list, rd2, reg2 = read_dicom(patientTwo)

            if not rs2_list:
                print("❌ No RTSTRUCT files found for patientTwo.")
                exit()

            # Build UID maps for patientTwo
            ct_by_uid2, pet_by_uid2 = {}, {}
            for ds in ct2:
                uid = getattr(ds, 'FrameOfReferenceUID', None)
                if uid:
                    ct_by_uid2.setdefault(uid, []).append(ds)
            for ds in pet2:
                uid = getattr(ds, 'FrameOfReferenceUID', None)
                if uid:
                    pet_by_uid2.setdefault(uid, []).append(ds)

            # RTSTRUCT chooser (patientTwo)
            valid_rs_options2, valid_rs_options_print2 = [], []
            print("\nMultiple RTSTRUCT files found for second patient:")
            for rs in rs2_list:
                uid2 = getattr(rs, "FrameOfReferenceUID", "UNKNOWN_UID")
                patient_name2 = getattr(rs, "PatientName", "Unknown")
                patient_id2 = getattr(rs, "PatientID", "Unknown")
                study_desc2 = getattr(rs, "StudyDescription", "No Description")
                modality2 = getattr(rs, "Modality", "RTSTRUCT")

                try:
                    rt2 = dicomparser.DicomParser(rs)
                    structures2 = rt2.GetStructures()
                    struct_names2 = ", ".join([s["name"] for s in structures2.values()])
                except Exception:
                    struct_names2 = "N/A"

                ct_match2  = "✅" if uid2 in ct_by_uid2  else "❌"
                pet_match2 = "✅" if uid2 in pet_by_uid2 else "❌"

                label2  = f"[{len(valid_rs_options2)}] {modality2} for {patient_name2} (ID: {patient_id2})"
                label2 += f"\n ├─ Study: {study_desc2}\n ├─ UID: {uid2}\n ├─ Linked CT: {ct_match2}, PET: {pet_match2}"
                label2 += f"\n └─ Structures:\n     {struct_names2}\n"

                valid_rs_options_print2.append(label2)
                valid_rs_options2.append((rs, uid2))

            if not valid_rs_options2:
                print("❌ No valid RTSTRUCTs with matching images for patientTwo.")
                exit()

            if len(valid_rs_options2) == 1:
                print("⚠️ Only one RTSTRUCT available; using it.")
                selected2 = 0
            else:
                selected2 = choose_rtstruct_gui(valid_rs_options2, valid_rs_options_print2)
                if selected2 is None:
                    print("❌ Invalid RTSTRUCT selection or user cancelled.")
                    exit()

            rs2, selected_uid2 = valid_rs_options2[selected2]
            matched_ct2  = ct_by_uid2.get(selected_uid2, [])
            matched_pet2 = pet_by_uid2.get(selected_uid2, [])

            if not matched_ct2 and not matched_pet2:
                print("❌ Selected RTSTRUCT (patientTwo) is not linked to any CT or PET series.")
                exit()

            print(f"\n✅ Using RTSTRUCT (patientTwo) UID={selected_uid2}")
            print(f"  -> CT slices: {len(matched_ct2)}")
            print(f"  -> PET slices: {len(matched_pet2)}")

            # Build previous timepoint for viewer (no extra boolean arg)
            prev_pet_volume, prev_masks, prev_structs = process_pvh(matched_ct2, matched_pet2, rd2, reg2, rs2, True)
        except Exception as _e:
            print("[Compare] Could not load previous timepoint volume for viewer:", _e)

        # Always run the CSV/plot diff using the folders you provided
        compare_timepoints_interactive("generated_data", "generated_data2",
                            curr_pet_volume=curr_pet_volume,
                            structure_masks=curr_masks,
                            structures=structures,
                            prev_pet_volume=prev_pet_volume)  # add this when you have it
        
    else:
        print("[Compare] Skipping disk-based comparison; no compare_data_dir or user chose No.")