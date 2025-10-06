import pydicom
import glob
import os
from config import elastix_file
from collections import defaultdict
from helper.resample import Volume, parse_rt_registration

""" 
Loads all of the DICOM Data and returns them as 
ct_datasets, pet_datasets, rs_dataset, rd_dataset, reg_dataset
"""

def read_dicom(patientFold):
    ct_series = defaultdict(list)
    pet_datasets = []
    rs_datasets = []
    rd_dataset = None
    reg_dataset = None

    dicom_files = glob.glob(os.path.join(patientFold, '**', '*'), recursive=True)
    for file_path in dicom_files:
        if not os.path.isfile(file_path):
            continue
        try:
            ds = pydicom.dcmread(file_path)
            modality = getattr(ds, 'Modality', None)
            if not modality:
                continue
            print(f"{os.path.basename(file_path)}: Modality = {modality}")

            if modality == 'CT':
                series_uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
                ct_series[series_uid].append(ds)
            elif modality == 'PT':
                pet_datasets.append(ds)
            elif modality == 'RTSTRUCT':
                rs_datasets.append(ds)
            elif modality == 'RTDOSE':
                rd_dataset = ds
            elif modality == 'REG':
                reg_dataset = ds

        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    print("\nLoad Summary")
    for uid, series in ct_series.items():
        print(f"  CT Series {uid}: {len(series)} slices")
    print(f"Loaded {len(pet_datasets)} PET slices")
    print("\nRTSTRUCT loaded - PASS - Ready" if rs_datasets else "\nRTSTRUCT loaded - FAIL")
    print("\nRTDOSE loaded - PASS - Ready" if rd_dataset else "\nRTDOSE loaded - FAIL")
    print("\nREG loaded - PASS - Ready" if reg_dataset else "\nREG loaded - FAIL (will use Elastix if needed)")

    os.makedirs("generated_data", exist_ok=True)
    os.makedirs("generated_data/DVH", exist_ok=True)
    os.makedirs("generated_data/PVH", exist_ok=True)

    # Handle 2 CT series with REG or Elastix fallback
    if len(ct_series) == 2:
        ct_series_list = list(ct_series.values())
        ct1_series = ct_series_list[0]
        ct2_series = ct_series_list[1]
        ct1_uid = ct1_series[0].FrameOfReferenceUID
        ct2_uid = ct2_series[0].FrameOfReferenceUID

        print(f"CT1 UID: {ct1_uid}")
        print(f"CT2 UID: {ct2_uid}")

        reg_transform = None
        should_register = False

        if reg_dataset:
            reg_transform = parse_rt_registration(reg_dataset, ct2_uid, ct1_uid)
            if reg_transform:
                print("✅ REG dataset matches CT2 → CT1 — proceeding with registration")
                should_register = True
        else:
            print("⚠️ No valid REG match for CT2 → CT1 — skipping registration")

        if should_register:

            ct_series_list = list(ct_series.values())
            uids = [series[0].FrameOfReferenceUID for series in ct_series_list]
            
            # Determine which is CT1 (fixed) and CT2 (moving)
            # Typically CT1 has the earlier acquisition time or is the reference
            ct1_series = ct_series_list[0]  # Fixed reference
            ct2_series = ct_series_list[1]  # Moving to be registered
            
            print(f"CT1 (fixed reference): {len(ct1_series)} slices, FrameOfReferenceUID: {ct1_series[0].FrameOfReferenceUID}")
            print(f"CT2 (moving): {len(ct2_series)} slices, FrameOfReferenceUID: {ct2_series[0].FrameOfReferenceUID}")

            try:
                volume = Volume()
                
                # Create volumes for both CT series
                ct1_volume, ct1_sorted = volume.create_ct_volume(ct1_series)
                ct2_volume, ct2_sorted = volume.create_ct_volume(ct2_series)
                
                if ct1_volume is None or ct2_volume is None:
                    print("Error: Invalid CT volume data")
                    ct_datasets = [ds for s in ct_series.values() for ds in s]
                else:
                    print(f"CT1 volume shape: {ct1_volume.shape}")
                    print(f"CT2 volume shape: {ct2_volume.shape}")
                    
                    # Register CT2 to CT1 coordinate system
                    print("Registering CT2 to CT1...")
                    resampled_ct2 = volume.resample_ct_to_ct(
                        ct2_series, ct1_series, ct2_volume,
                        reg_transform=reg_transform,
                        use_external_elastix=True if not reg_transform else False,
                        elastix_param_file=elastix_file
                    )
                    
                    print(f"CT2 registration completed. Output shape: {resampled_ct2.shape}")
                    
                    # Use CT1 as the reference coordinate system
                    ct_datasets = ct1_series
                    
            except Exception as e:
                print(f"CT registration failed: {e}")
                print("Using all CT datasets without registration")
                ct_datasets = [ds for s in ct_series.values() for ds in s]
        else:
            # Only one CT series or registration not needed
            print(f"Found {len(ct_series)} CT series. No registration needed.")
            ct_datasets = [ds for s in ct_series.values() for ds in s]
    else:
        # Only one CT series or registration not needed
        print(f"Found {len(ct_series)} CT series. No registration needed.")
        ct_datasets = [ds for s in ct_series.values() for ds in s]
        
    return ct_datasets, pet_datasets, rs_datasets, rd_dataset, reg_dataset
