from dicompylercore import dicomparser

def groupDataUID(ct_datasets, pet_datasets):
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
    
    return ct_by_uid, pet_by_uid
    

def getValidRS(rs_datasets, ct_by_uid, pet_by_uid):
    valid_rs_options = []
    valid_rs_options_print = []
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
    
    return valid_rs_options, valid_rs_options_print

