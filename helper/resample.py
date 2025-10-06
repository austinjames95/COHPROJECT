import SimpleITK as sitk
import numpy as np
import os
import subprocess
from config import elastix_output

def parse_rt_registration(reg_dataset, ct_from_uid, ct_to_uid):
    """Parse RT registration dataset and return transformation if found"""
    if not reg_dataset or reg_dataset.Modality != "REG":
        print("Invalid or missing REG dataset")
        return None

    try:
        reg_items = reg_dataset.RegistrationSequence
    except AttributeError:
        print("No RegistrationSequence found in REG dataset")
        return None

    print(f"Found {len(reg_items)} registration items in REG dataset")
    
    for i, item in enumerate(reg_items):
        print(f"Processing registration item {i}")
        
        # Check if this registration item matches our source frame
        from_uid = getattr(item, 'FrameOfReferenceUID', None)
        print(f"  From UID: {from_uid}")
        print(f"  Looking for: {ct_from_uid}")
        
        if from_uid != ct_from_uid:
            print(f"  Frame of reference UID mismatch, skipping item {i}")
            continue

        try:
            # Check if MatrixRegistrationSequence exists
            if not hasattr(item, 'MatrixRegistrationSequence') or len(item.MatrixRegistrationSequence) == 0:
                print(f"  No MatrixRegistrationSequence in item {i}")
                continue
                
            matrix_seq = item.MatrixRegistrationSequence[0]
            
            # Check for the transformation matrix
            if not hasattr(matrix_seq, 'FrameOfReferenceTransformationMatrix'):
                print(f"  Missing FrameOfReferenceTransformationMatrix in item {i}")
                continue
                
            matrix = matrix_seq.FrameOfReferenceTransformationMatrix
            print(f"  Found transformation matrix with {len(matrix)} elements")
            
        except (AttributeError, IndexError) as e:
            print(f"  Error accessing MatrixRegistrationSequence in item {i}: {e}")
            continue

        if len(matrix) != 16:
            print(f"  Unexpected matrix length: {len(matrix)} (expected 16)")
            continue

        try:
            # Convert to 4x4 transformation matrix
            transform_matrix = np.array(matrix, dtype=np.float64).reshape(4, 4)
            print(f"  Transformation matrix:\n{transform_matrix}")
            
            # Create SimpleITK affine transform
            affine = sitk.AffineTransform(3)
            
            # Set the 3x3 rotation/scaling matrix (flattened)
            affine.SetMatrix(transform_matrix[:3, :3].flatten())
            
            # Set the translation vector
            affine.SetTranslation(transform_matrix[:3, 3].tolist())
            
            print(f"  Successfully created affine transform")
            return affine
            
        except Exception as e:
            print(f"  Error creating transform from matrix in item {i}: {e}")
            continue

    print("No valid transformation found in REG dataset")
    return None

class RegistrationManager:
    def __init__(self):
        self.registration_transform = None
        self.elastix_available = self._check_elastix()

    def _check_elastix(self):
        """Check if Elastix is available through SimpleITK"""
        try:
            sitk.ElastixImageFilter()
            print("SimpleITK Elastix available")
            return True
        except:
            # Check if external elastix command is available
            try:
                result = subprocess.run(['elastix', '--help'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print("External Elastix command available")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            print("Warning: Elastix not available. Manual registration transforms required.")
            return False

    def run_elastix_external(self, fixed_path, moving_path, output_dir, param_file):
        """Run external elastix command"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Verify parameter file exists
        if not os.path.exists(param_file):
            raise FileNotFoundError(f"Parameter file not found: {param_file}")
            
        cmd = ["elastix", "-f", fixed_path, "-m", moving_path, "-out", output_dir, "-p", param_file]
        print("Running external Elastix command:\n", " ".join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Elastix stdout:\n", result.stdout)
            print("Elastix stderr:\n", result.stderr)
            raise RuntimeError(f"Elastix failed with return code {result.returncode}")
        
        print("Elastix completed successfully")

    def auto_register_with_elastix(self, pet_img, ct_img, use_external=False, param_file=None, output_dir="elastix_output"):
        """Perform automatic registration using Elastix"""
        if use_external:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            fixed_path = os.path.join(output_dir, "fixed.nii.gz")
            moving_path = os.path.join(output_dir, "moving.nii.gz")
            
            # Save images for external processing
            sitk.WriteImage(ct_img, fixed_path)
            sitk.WriteImage(pet_img, moving_path)

            if param_file is None:
                raise ValueError("Parameter file required for external Elastix")

            self.run_elastix_external(fixed_path, moving_path, output_dir, param_file)
            
            result_path = os.path.join(output_dir, "result.0.nii.gz")
            if not os.path.exists(result_path):
                raise FileNotFoundError("Expected output file from Elastix not found: result.0.nii.gz")

            print("External registration completed. Reading result image...")
            return sitk.ReadImage(result_path)
        else:
            # Use SimpleITK Elastix
            parameter_map = sitk.GetDefaultParameterMap("rigid")
            parameter_map["NumberOfResolutions"] = ["3"]
            parameter_map["MaximumNumberOfIterations"] = ["500"]
            parameter_map["SP_alpha"] = ["0.6"]

            elastix_filter = sitk.ElastixImageFilter()
            elastix_filter.SetFixedImage(ct_img)
            elastix_filter.SetMovingImage(pet_img)
            elastix_filter.SetParameterMap(parameter_map)
            elastix_filter.Execute()

            transform_map = elastix_filter.GetTransformParameterMap()[0]
            self.registration_transform = sitk.ReadTransform(transform_map)

            return elastix_filter.GetResultImage()

    def apply_manual_transform(self, transform_matrix=None, translation=None, rotation=None):
        """Apply manual transformation parameters"""
        if transform_matrix is not None:
            if len(transform_matrix) == 12:
                transform = sitk.AffineTransform(3)
                transform.SetMatrix(transform_matrix[:9])
                transform.SetTranslation(transform_matrix[9:12])
            else:
                raise ValueError("Transform matrix must have 12 elements")
        else:
            transform = sitk.Euler3DTransform()
            if rotation is not None:
                transform.SetRotation(*rotation)
            if translation is not None:
                transform.SetTranslation(translation)

        self.registration_transform = transform
        return transform

class Volume:
    def __init__(self):
        self.registration_manager = RegistrationManager()

    def create_ct_volume(self, ct_datasets):
        """Create CT volume from DICOM datasets"""
        if not ct_datasets:
            print("Error: No CT datasets provided")
            return None, []
            
        ct_sorted = sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
        rows = ct_sorted[0].Rows
        cols = ct_sorted[0].Columns
        slices = len(ct_sorted)

        print(f"Creating CT volume: {slices} slices, {rows}x{cols} pixels")
        ct_volume = np.zeros((slices, rows, cols), dtype=np.float32)

        for i, ds in enumerate(ct_sorted):
            try:
                array = ds.pixel_array.astype(np.float32)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    array = array * ds.RescaleSlope + ds.RescaleIntercept
                ct_volume[i] = array
            except Exception as e:
                print(f"Error reading CT slice {i}: {e}")
                ct_volume[i] = np.zeros((rows, cols))

        return ct_volume, ct_sorted
    
    def create_pet_volume(self, pet_datasets):
        """Create PET volume from DICOM datasets"""
        if not pet_datasets:
            print("Error: No PET datasets provided")
            return None, []
            
        pet_sorted = sorted(pet_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
        rows = pet_sorted[0].Rows
        cols = pet_sorted[0].Columns
        slices = len(pet_sorted)

        print(f"Creating PET volume: {slices} slices, {rows}x{cols} pixels")
        pet_volume = np.zeros((slices, rows, cols), dtype=np.float32)

        for i, ds in enumerate(pet_sorted):
            try:
                array = ds.pixel_array.astype(np.float32)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    array = array * ds.RescaleSlope + ds.RescaleIntercept
                pet_volume[i] = array
            except Exception as e:
                print(f"Error reading PET slice {i}: {e}")
                pet_volume[i] = np.zeros((rows, cols))

        return pet_volume, pet_sorted

    def resample_ct_to_ct(self, moving_ct_datasets, fixed_ct_datasets, moving_ct_volume,
                          reg_transform=None, use_external_elastix=False,
                          elastix_param_file=None, elastix_output_dir=None):
        """Resample moving CT to match fixed CT coordinate system"""
        
        if elastix_output_dir is None:
            elastix_output_dir = elastix_output
            
        print("Starting CT-to-CT resampling...")
        
        # Validate inputs
        if not moving_ct_datasets or not fixed_ct_datasets:
            raise ValueError("Both moving and fixed CT datasets are required")
            
        if moving_ct_volume is None:
            raise ValueError("Moving CT volume is required")

        # Prepare moving CT image
        moving_sorted = sorted(moving_ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
        moving_spacing = list(map(float, moving_sorted[0].PixelSpacing))
        
        # Calculate slice spacing
        if len(moving_sorted) > 1:
            dz_moving = abs(float(moving_sorted[1].ImagePositionPatient[2]) - float(moving_sorted[0].ImagePositionPatient[2]))
        else:
            dz_moving = float(moving_sorted[0].SliceThickness) if hasattr(moving_sorted[0], 'SliceThickness') else 1.0
            
        moving_spacing.append(dz_moving)
        moving_origin = list(map(float, moving_sorted[0].ImagePositionPatient))
        
        # Calculate direction matrix for moving image
        iop = moving_sorted[0].ImageOrientationPatient
        axis_x = np.array(iop[:3])
        axis_y = np.array(iop[3:])
        axis_z = np.cross(axis_x, axis_y)
        moving_direction = np.concatenate([axis_x, axis_y, axis_z]).tolist()

        moving_img = sitk.GetImageFromArray(moving_ct_volume.astype(np.float32))
        moving_img.SetSpacing([moving_spacing[1], moving_spacing[0], moving_spacing[2]])  # Note: x,y swapped for SimpleITK
        moving_img.SetOrigin(moving_origin)
        moving_img.SetDirection(moving_direction)

        # Prepare fixed CT reference
        fixed_sorted = sorted(fixed_ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
        fixed_shape = (fixed_sorted[0].Columns, fixed_sorted[0].Rows, len(fixed_sorted))
        fixed_spacing = list(map(float, fixed_sorted[0].PixelSpacing))
        
        if len(fixed_sorted) > 1:
            dz_fixed = abs(float(fixed_sorted[1].ImagePositionPatient[2]) - float(fixed_sorted[0].ImagePositionPatient[2]))
        else:
            dz_fixed = float(fixed_sorted[0].SliceThickness) if hasattr(fixed_sorted[0], 'SliceThickness') else 1.0
            
        fixed_spacing.append(dz_fixed)
        fixed_origin = list(map(float, fixed_sorted[0].ImagePositionPatient))
        
        iop_fixed = fixed_sorted[0].ImageOrientationPatient
        axis_x_fixed = np.array(iop_fixed[:3])
        axis_y_fixed = np.array(iop_fixed[3:])
        axis_z_fixed = np.cross(axis_x_fixed, axis_y_fixed)
        fixed_direction = np.concatenate([axis_x_fixed, axis_y_fixed, axis_z_fixed]).tolist()

        fixed_ref = sitk.Image(fixed_shape, sitk.sitkFloat32)
        fixed_ref.SetSpacing([fixed_spacing[1], fixed_spacing[0], fixed_spacing[2]])
        fixed_ref.SetOrigin(fixed_origin)
        fixed_ref.SetDirection(fixed_direction)

        # Apply CT-to-CT registration transform
        if reg_transform is not None:
            print("Using provided registration transform for CT-to-CT registration")
            final_transform = reg_transform
            
        elif use_external_elastix and self.registration_manager.elastix_available:
            print("Using Elastix for CT-to-CT registration")
            try:
                if elastix_param_file is None:
                    raise ValueError("External Elastix requires a parameter file")

                # Create fixed CT image for registration
                fixed_volume, _ = self.create_ct_volume(fixed_ct_datasets)
                fixed_img = sitk.GetImageFromArray(fixed_volume.astype(np.float32))
                fixed_img.SetSpacing([fixed_spacing[1], fixed_spacing[0], fixed_spacing[2]])
                fixed_img.SetOrigin(fixed_origin)
                fixed_img.SetDirection(fixed_direction)

                registered_img = self.registration_manager.auto_register_with_elastix(
                    moving_img, fixed_img, use_external=True, 
                    param_file=elastix_param_file, output_dir=elastix_output_dir
                )
                
                result_array = sitk.GetArrayFromImage(registered_img)
                print(f"CT-to-CT registration completed using Elastix. Output shape: {result_array.shape}")
                return result_array
                
            except Exception as e:
                print(f"Elastix CT-to-CT registration failed: {e}")
                print("Falling back to identity transform")
                final_transform = sitk.Transform(3, sitk.sitkIdentity)
        else:
            print("Using identity transform for CT-to-CT registration")
            final_transform = sitk.Transform(3, sitk.sitkIdentity)

        # Apply final resampling
        print("Applying CT-to-CT resampling...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_ref)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000)  # Typical air value for CT
        resampler.SetTransform(final_transform)

        resampled_ct = resampler.Execute(moving_img)
        self.registration_manager.registration_transform = final_transform
        
        result_array = sitk.GetArrayFromImage(resampled_ct)
        print(f"CT-to-CT resampling completed. Output shape: {result_array.shape}")
        
        return result_array

    def resample_pet_to_ct(self, pet_datasets, ct_datasets, pet_volume,
                           manual_transform=None, use_auto_registration=True,
                           ct_volume=None, use_external_elastix=False,
                           elastix_param_file=None, elastix_output_dir=None,
                           reg_dataset=None):
            
        print("Starting PET to CT resampling...")
        
        # Validate inputs
        if not pet_datasets or not ct_datasets:
            raise ValueError("Both PET and CT datasets are required")
            
        if pet_volume is None:
            raise ValueError("PET volume is required")

        # Prepare PET image
        pet_sorted = sorted(pet_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
        pet_spacing = list(map(float, pet_sorted[0].PixelSpacing))
        
        # Calculate slice spacing
        if len(pet_sorted) > 1:
            dz_pet = abs(float(pet_sorted[1].ImagePositionPatient[2]) - float(pet_sorted[0].ImagePositionPatient[2]))
        else:
            dz_pet = float(pet_sorted[0].SliceThickness) if hasattr(pet_sorted[0], 'SliceThickness') else 1.0
            
        pet_spacing.append(dz_pet)
        pet_origin = list(map(float, pet_sorted[0].ImagePositionPatient))
        
        # Calculate direction matrix
        iop = pet_sorted[0].ImageOrientationPatient
        axis_x = np.array(iop[:3])
        axis_y = np.array(iop[3:])
        axis_z = np.cross(axis_x, axis_y)
        direction = np.concatenate([axis_x, axis_y, axis_z]).tolist()

        pet_img = sitk.GetImageFromArray(pet_volume.astype(np.float32))
        pet_img.SetSpacing([pet_spacing[1], pet_spacing[0], pet_spacing[2]])  # Note: x,y swapped for SimpleITK
        pet_img.SetOrigin(pet_origin)
        pet_img.SetDirection(direction)

        # Prepare CT reference
        ct_sorted = sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
        ct_shape = (ct_sorted[0].Columns, ct_sorted[0].Rows, len(ct_sorted))
        ct_spacing = list(map(float, ct_sorted[0].PixelSpacing))
        
        if len(ct_sorted) > 1:
            dz_ct = abs(float(ct_sorted[1].ImagePositionPatient[2]) - float(ct_sorted[0].ImagePositionPatient[2]))
        else:
            dz_ct = float(ct_sorted[0].SliceThickness) if hasattr(ct_sorted[0], 'SliceThickness') else 1.0
            
        ct_spacing.append(dz_ct)
        ct_origin = list(map(float, ct_sorted[0].ImagePositionPatient))
        
        iop_ct = ct_sorted[0].ImageOrientationPatient
        axis_x_ct = np.array(iop_ct[:3])
        axis_y_ct = np.array(iop_ct[3:])
        axis_z_ct = np.cross(axis_x_ct, axis_y_ct)
        ct_direction = np.concatenate([axis_x_ct, axis_y_ct, axis_z_ct]).tolist()

        ct_ref = sitk.Image(ct_shape, sitk.sitkFloat32)
        ct_ref.SetSpacing([ct_spacing[1], ct_spacing[0], ct_spacing[2]])
        ct_ref.SetOrigin(ct_origin)
        ct_ref.SetDirection(ct_direction)

        # Apply CT-to-CT transform if REG dataset is available
        if reg_dataset and hasattr(ct_sorted[0], 'FrameOfReferenceUID'):
            try:
                # Check if we need to apply CT-to-CT registration
                from_uid = ct_sorted[0].FrameOfReferenceUID
                
                # Get target frame of reference from REG dataset
                if hasattr(reg_dataset, 'ReferencedFrameOfReferenceSequence') and len(reg_dataset.ReferencedFrameOfReferenceSequence) > 0:
                    to_uid = reg_dataset.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                    
                    if from_uid != to_uid:
                        print(f"Applying CT-to-CT transform: {from_uid} -> {to_uid}")
                        reg_transform = parse_rt_registration(reg_dataset, from_uid, to_uid)
                        if reg_transform:
                            print("Applying CT-to-CT transform from REG.dcm")
                            resampler_ct = sitk.ResampleImageFilter()
                            resampler_ct.SetReferenceImage(ct_ref)
                            resampler_ct.SetInterpolator(sitk.sitkLinear)
                            resampler_ct.SetTransform(reg_transform)
                            resampler_ct.SetDefaultPixelValue(0)
                            ct_ref = resampler_ct.Execute(ct_ref)
                        else:
                            print("Could not extract valid transform from REG dataset")
                    else:
                        print("Frame of reference UIDs match, no CT-to-CT transform needed")
                else:
                    print("No ReferencedFrameOfReferenceSequence in REG dataset")
            except Exception as e:
                print(f"Error processing REG dataset for CT-to-CT transform: {e}")

        # PET-to-CT registration
        reg_transform = None
        
        if manual_transform is not None:
            print("Using manual registration transform")
            reg_transform = self.registration_manager.apply_manual_transform(transform_matrix=manual_transform)
            
        elif use_auto_registration and self.registration_manager.elastix_available:
            print("Attempting auto-registration with Elastix")
            try:
                if ct_volume is None:
                    ct_volume, _ = self.create_ct_volume(ct_datasets)
                    
                ct_img = sitk.GetImageFromArray(ct_volume.astype(np.float32))
                ct_img.SetSpacing([ct_spacing[1], ct_spacing[0], ct_spacing[2]])
                ct_img.SetOrigin(ct_origin)
                ct_img.SetDirection(ct_direction)
                
                if use_external_elastix:
                    if elastix_param_file is None:
                        raise ValueError("External Elastix requires a parameter file")

                    registered_img = self.registration_manager.auto_register_with_elastix(
                        pet_img, ct_img, use_external=True, 
                        param_file=elastix_param_file, output_dir=elastix_output_dir
                    )
                    return sitk.GetArrayFromImage(registered_img)
                else:
                    registered_img = self.registration_manager.auto_register_with_elastix(pet_img, ct_img)
                    return sitk.GetArrayFromImage(registered_img)
                    
            except Exception as e:
                print(f"Auto-registration failed: {e}")
                print("Falling back to identity transform")
                reg_transform = sitk.Transform(3, sitk.sitkIdentity)
        else:
            print("Using identity transform (no registration)")
            reg_transform = sitk.Transform(3, sitk.sitkIdentity)

        # Apply final resampling
        print("Applying final resampling...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_ref)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(reg_transform)

        resampled_pet = resampler.Execute(pet_img)
        self.registration_manager.registration_transform = reg_transform
        
        result_array = sitk.GetArrayFromImage(resampled_pet)
        print(f"Resampling completed. Output shape: {result_array.shape}")
        
        return result_array

    def get_registration_transform(self):
        """Get the last computed registration transform"""
        return self.registration_manager.registration_transform
