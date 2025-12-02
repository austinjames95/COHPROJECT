import numpy as np
import os
from scipy import ndimage
from skimage.measure import label
from rt_utils import RTStructBuilder
import shutil
import csv

def find_largest_connected_component(binary_mask):
    """Find the largest connected component in a binary mask (memory-efficient)."""
    labeled_mask = label(binary_mask, connectivity=3)  # 3D connectivity

    if labeled_mask.max() == 0:
        return binary_mask

    # Get unique labels and counts (avoids massive allocation in np.bincount)
    labels, counts = np.unique(labeled_mask, return_counts=True)
    
    # Remove background (label 0)
    labels = labels[1:]
    counts = counts[1:]
    
    if len(counts) == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    largest_label = labels[np.argmax(counts)]
    return labeled_mask == largest_label

def create_suv_threshold_structure(suv_volume, reference_roi_mask, threshold_percent=50, 
                                 connected_only=True, min_volume_ml=0.1, voxel_volume_ml=1.0):
    """
    Create a structure based on SUV threshold within a reference ROI.
    
    Parameters:
    - suv_volume: 3D SUV volume
    - reference_roi_mask: 3D boolean mask of reference ROI
    - threshold_percent: Percentage of max SUV to use as threshold (default 50%)
    - connected_only: If True, keep only largest connected component
    - min_volume_ml: Minimum volume in mL to keep structure
    - voxel_volume_ml: Volume of each voxel in mL
    
    Returns:
    - new_mask: 3D boolean mask of the new structure
    - stats: Dictionary with statistics about the new structure
    """
    if not np.any(reference_roi_mask):
        print("Warning: Reference ROI mask is empty")
        return np.zeros_like(suv_volume, dtype=bool), {}
    
    # Get SUV values within the reference ROI
    roi_suv_values = suv_volume[reference_roi_mask]
    
    if roi_suv_values.size == 0:
        print("Warning: No SUV values found in reference ROI")
        return np.zeros_like(suv_volume, dtype=bool), {}
    
    # Calculate threshold
    max_suv = np.max(roi_suv_values)
    threshold_suv = max_suv * (threshold_percent / 100.0)
    
    print(f"Max SUV in ROI: {max_suv:.3f}")
    print(f"Threshold ({threshold_percent}%): {threshold_suv:.3f}")
    
    # Create threshold mask within reference ROI
    threshold_mask = reference_roi_mask & (suv_volume >= threshold_suv)
    
    if not np.any(threshold_mask):
        print(f"Warning: No voxels above {threshold_percent}% threshold")
        return np.zeros_like(suv_volume, dtype=bool), {}
    
    # Keep only largest connected component if requested
    if connected_only:
        threshold_mask = find_largest_connected_component(threshold_mask)
    
    # Check minimum volume
    volume_ml = np.sum(threshold_mask) * voxel_volume_ml
    if volume_ml < min_volume_ml:
        print(f"Warning: Generated structure volume ({volume_ml:.3f} mL) below minimum ({min_volume_ml} mL)")
        return np.zeros_like(suv_volume, dtype=bool), {}
    
    # Calculate statistics
    new_suv_values = suv_volume[threshold_mask]
    stats = {
        'volume_ml': volume_ml,
        'voxel_count': np.sum(threshold_mask),
        'suv_mean': np.mean(new_suv_values),
        'suv_max': np.max(new_suv_values),
        'suv_min': np.min(new_suv_values),
        'threshold_suv': threshold_suv,
        'threshold_percent': threshold_percent
    }
    
    return threshold_mask, stats

def create_multiple_threshold_structures(suv_volume, reference_roi_mask, 
                                       thresholds=[90, 70, 50, 30], 
                                       connected_only=True, min_volume_ml=0.1, 
                                       voxel_volume_ml=1.0):
    """
    Create multiple structures at different SUV thresholds.
    
    Returns:
    - structures: Dictionary with threshold as key and (mask, stats) as value
    """
    structures = {}
    
    for threshold in thresholds:
        mask, stats = create_suv_threshold_structure(
            suv_volume, reference_roi_mask, threshold, 
            connected_only, min_volume_ml, voxel_volume_ml
        )
        
        if np.any(mask):
            structures[threshold] = {'mask': mask, 'stats': stats}
            print(f"Created {threshold}% threshold structure: {stats['volume_ml']:.3f} mL")
        else:
            print(f"Skipped {threshold}% threshold structure (empty or too small)")
    
    return structures

def create_gradient_structure(suv_volume, reference_roi_mask, erosion_mm=2, 
                            voxel_spacing_mm=[1.0, 1.0, 1.0], min_volume_ml=0.1, 
                            voxel_volume_ml=1.0):
    """
    Create a structure representing the high-gradient region (tumor edge).
    
    Parameters:
    - erosion_mm: How much to erode the reference ROI in mm
    - voxel_spacing_mm: Voxel spacing in [x, y, z] mm
    """
    if not np.any(reference_roi_mask):
        return np.zeros_like(suv_volume, dtype=bool), {}
    
    # Convert erosion from mm to voxels
    erosion_voxels = [int(erosion_mm / spacing) for spacing in voxel_spacing_mm]
    
    # Create eroded mask
    structure_element = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    eroded_mask = reference_roi_mask.copy()
    
    for i, erosion in enumerate(erosion_voxels):
        if erosion > 0:
            # Apply erosion in each dimension
            eroded_mask = ndimage.binary_erosion(eroded_mask, structure_element, iterations=erosion)
    
    # Gradient structure is original minus eroded (rim/edge region)
    gradient_mask = reference_roi_mask & ~eroded_mask
    
    volume_ml = np.sum(gradient_mask) * voxel_volume_ml
    if volume_ml < min_volume_ml:
        print(f"Warning: Gradient structure volume ({volume_ml:.3f} mL) below minimum")
        return np.zeros_like(suv_volume, dtype=bool), {}
    
    # Calculate statistics
    if np.any(gradient_mask):
        gradient_suv_values = suv_volume[gradient_mask]
        stats = {
            'volume_ml': volume_ml,
            'voxel_count': np.sum(gradient_mask),
            'suv_mean': np.mean(gradient_suv_values),
            'suv_max': np.max(gradient_suv_values),
            'suv_min': np.min(gradient_suv_values),
            'erosion_mm': erosion_mm
        }
    else:
        stats = {}
    
    return gradient_mask, stats

def resample_mask_to_ct_shape(mask, target_shape):
    """
    Resample a 3D boolean mask to match target CT shape.
    
    Parameters:
    - mask: 3D boolean array to resample
    - target_shape: Tuple (z, y, x) of target dimensions
    
    Returns:
    - resampled_mask: Boolean mask with target_shape
    """
    
    print(f"    Resampling mask from {mask.shape} to {target_shape}")
    
    # Calculate zoom factors for each dimension
    zoom_factors = [
        target_shape[0] / mask.shape[0],  # z
        target_shape[1] / mask.shape[1],  # y  
        target_shape[2] / mask.shape[2]   # x
    ]
    
    print(f"    Zoom factors: {zoom_factors}")
    
    # Convert to float for interpolation
    mask_float = mask.astype(np.float32)
    
    # Use scipy.ndimage.zoom with nearest neighbor interpolation
    resampled_float = ndimage.zoom(
        mask_float, 
        zoom_factors, 
        order=0,  # Nearest neighbor interpolation
        mode='constant',
        cval=0.0,
        prefilter=False  # Important for binary data
    )
    
    print(f"    After zoom: {resampled_float.shape}")
    
    # Convert back to boolean
    resampled_mask = resampled_float > 0.5
    
    # Ensure exact target shape (scipy zoom might be off by 1 voxel due to rounding)
    if resampled_mask.shape != target_shape:
        print(f"    Adjusting shape from {resampled_mask.shape} to {target_shape}")
        
        # Create output array with exact target shape
        final_mask = np.zeros(target_shape, dtype=bool)
        
        # Calculate the region to copy (handle both cropping and padding)
        copy_shape = tuple(min(resampled_mask.shape[i], target_shape[i]) for i in range(3))
        
        # Copy the overlapping region
        final_mask[:copy_shape[0], :copy_shape[1], :copy_shape[2]] = \
            resampled_mask[:copy_shape[0], :copy_shape[1], :copy_shape[2]]
        
        resampled_mask = final_mask
    
    print(f"    Final resampled shape: {resampled_mask.shape}")
    
    # Verify the shape is exactly what we want
    assert resampled_mask.shape == target_shape, f"Resampling failed: got {resampled_mask.shape}, expected {target_shape}"
    
    return resampled_mask

def write_structures_to_rt(reference_ct_datasets, new_structures, 
                          output_rs_path):
    """
    Write new structures to an RT Structure Set file using rt-utils.
    
    Parameters:
    - reference_ct_datasets: List of CT DICOM datasets for reference
    - original_rs_path: Path to original RT Structure Set (optional, for copying existing structures)
    - new_structures: Dictionary of structure_name: {'mask': mask, 'color': [R,G,B], 'stats': stats}
    - output_rs_path: Output path for new RT Structure Set
    """
    if not reference_ct_datasets:
        raise ValueError("Reference CT datasets are required for RT Structure Set creation")
    
    temp_ct_dir = None
    try:
        # Create temporary directory for CT files
        temp_ct_dir = "temp_ct_for_rtstruct"
        os.makedirs(temp_ct_dir, exist_ok=True)
        
        print("Preparing CT datasets for RT Structure Set creation...")
        
        # Sort CT datasets by SliceLocation to ensure proper ordering
        try:
            ct_datasets_sorted = sorted(reference_ct_datasets, 
                                      key=lambda x: float(x.SliceLocation))
            print("CT datasets sorted by SliceLocation")
        except:
            # If SliceLocation is not available, use InstanceNumber or original order
            try:
                ct_datasets_sorted = sorted(reference_ct_datasets,
                                          key=lambda x: int(x.InstanceNumber))
                print("CT datasets sorted by InstanceNumber")
            except:
                ct_datasets_sorted = reference_ct_datasets
                print("Using original CT dataset order")
        
        # Save CT datasets temporarily with proper naming
        ct_filenames = []
        for i, ct_ds in enumerate(ct_datasets_sorted):
            # Ensure proper DICOM file extension
            temp_filename = os.path.join(temp_ct_dir, f"CT_{i:04d}.dcm")
            ct_ds.save_as(temp_filename)
            ct_filenames.append(temp_filename)
        
        print(f"Saved {len(ct_filenames)} CT files to temporary directory")
        
        # Get CT dimensions for validation - RT-utils expects (slices, rows, cols)
        num_slices = len(ct_datasets_sorted)
        rows = ct_datasets_sorted[0].Rows
        cols = ct_datasets_sorted[0].Columns
        ct_shape = (num_slices, rows, cols)
        
        print(f"CT series shape (slices, rows, cols): {ct_shape}")
        print(f"First CT slice: {rows}x{cols}")
        
        # Create new RT struct from CT directory
        print("Creating RT Structure Set from CT series...")
        rtstruct = RTStructBuilder.create_new(dicom_series_path=temp_ct_dir)
        
        # Add each new structure with dimension validation
        print(f"Adding {len(new_structures)} structures to RT Structure Set...")
        successfully_added = 0
        
        for struct_name, struct_data in new_structures.items():
            mask = struct_data['mask']
            color = struct_data.get('color', [255, 0, 0])  # Default red
            
            print(f"  Processing: {struct_name}")
            print(f"    Original mask shape: {mask.shape}")
            print(f"    Required CT shape: {ct_shape}")
            print(f"    Volume: {struct_data['stats'].get('volume_ml', 0):.3f} mL")
            print(f"    Color: RGB{tuple(color)}")
            
            # Ensure mask is boolean
            if mask.dtype != bool:
                mask = mask.astype(bool)
            
            # Check if mask dimensions match CT dimensions
            if mask.shape != ct_shape:
                print(f"    ⚠️  Mask shape {mask.shape} doesn't match CT shape {ct_shape}")
                
                # Resample mask to match CT dimensions
                try:
                    mask = resample_mask_to_ct_shape(mask, ct_shape)
                    print(f"    ✓ Resampled mask to shape: {mask.shape}")
                except Exception as resample_error:
                    print(f"    ❌ Could not resample mask: {resample_error}")
                    print(f"    Skipping structure: {struct_name}")
                    continue
            
            # Verify mask has some content
            if not np.any(mask):
                print(f"    ❌ Mask is empty, skipping structure: {struct_name}")
                continue
            
            # Final shape verification before adding to RT struct
            if mask.shape != ct_shape:
                print(f"    ❌ Final mask shape {mask.shape} still doesn't match CT shape {ct_shape}")
                print(f"    Skipping structure: {struct_name}")
                continue
            
            # Debug: Check the mask dimensions in detail
            print(f"    Final mask check:")
            print(f"      Mask shape: {mask.shape} = (slices={mask.shape[0]}, rows={mask.shape[1]}, cols={mask.shape[2]})")
            print(f"      CT shape:   {ct_shape} = (slices={ct_shape[0]}, rows={ct_shape[1]}, cols={ct_shape[2]})")
            print(f"      Mask dtype: {mask.dtype}")
            print(f"      Mask has {np.sum(mask)} True voxels")
            
            # Add structure to RT struct
            try:
                # Ensure mask is contiguous in memory
                mask = np.ascontiguousarray(mask)
                
                rtstruct.add_roi(
                    mask=mask,
                    color=color,
                    name=struct_name
                )
                print(f"    ✅ Successfully added {struct_name}")
                successfully_added += 1
                
            except Exception as add_error:
                print(f"    ❌ Error adding {struct_name}: {add_error}")
                print(f"    Error type: {type(add_error).__name__}")
                

                if "must have the save number of layers" in str(add_error):
                    print(f"    🔄 Trying alternative dimension interpretation...")
                    
                    # Try transposing the mask - maybe RT-utils expects (rows, cols, slices)?
                    try:
                        # Current: (slices, rows, cols) -> Try: (cols, slices, rows)
                        mask_transposed = np.transpose(mask, (1, 2, 0))
                        print(f"    Trying transposed mask shape: {mask_transposed.shape}")
                        
                        rtstruct.add_roi(
                            mask=mask_transposed,
                            color=color,
                            name=struct_name
                        )
                        print(f"    ✅ Successfully added {struct_name} with transposed mask")
                        successfully_added += 1
                        continue
                        
                    except Exception as transpose_error:
                        print(f"    ❌ Transposed approach also failed: {transpose_error}")
                        
                        # Try another permutation: (rows, cols, slices) 
                        try:
                            mask_perm2 = np.transpose(mask, (2, 0, 1)) 
                            print(f"    Trying second permutation shape: {mask_perm2.shape}")
                            
                            rtstruct.add_roi(
                                mask=mask_perm2,
                                color=color,
                                name=struct_name
                            )
                            print(f"    ✅ Successfully added {struct_name} with second permutation")
                            successfully_added += 1
                            continue
                            
                        except Exception as perm2_error:
                            print(f"    ❌ Second permutation also failed: {perm2_error}")
                
                # Print detailed error info for debugging
                import traceback
                print(f"    Detailed error traceback:")
                traceback.print_exc()
                continue
        
        if successfully_added == 0:
            print("❌ No structures were successfully added to RT Structure Set")
            return
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_rs_path), exist_ok=True)
        
        # Save the RT structure set
        print(f"Saving RT Structure Set with {successfully_added} structures to: {output_rs_path}")
        rtstruct.save(output_rs_path)
        print("✅ RT Structure Set saved successfully")
        
    except Exception as e:
        print(f"❌ Error creating RT Structure Set: {e}")
        return
        
    finally:
        # Clean up temporary files
        if temp_ct_dir and os.path.exists(temp_ct_dir):
            try:
                print("Cleaning up temporary files...")
                shutil.rmtree(temp_ct_dir)
                print("✅ Temporary files cleaned up")
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temporary files: {e}")

def generate_pet_based_structures(suv_volume, structure_masks, structures, 
                                reference_structure_name, ct_datasets,
                                voxel_volume_ml=1.0, voxel_spacing_mm=[1.0, 1.0, 1.0]):
    """
    Generate multiple PET-based structures from a reference structure.
    
    Parameters:
    - suv_volume: 3D SUV volume
    - structure_masks: Dictionary of structure masks from original RT struct
    - structures: Dictionary of structure information
    - reference_structure_name: Name of reference structure to base new structures on
    - ct_datasets: CT datasets for RT struct creation
    - voxel_volume_ml: Volume per voxel in mL
    - voxel_spacing_mm: Voxel spacing in mm
    
    Returns:
    - new_structures: Dictionary ready for RT struct creation
    """
    # Find reference structure
    reference_sid = None
    reference_mask = None
    
    for sid, struct_info in structures.items():
        if struct_info['name'].lower() == reference_structure_name.lower():
            reference_sid = sid
            reference_mask = structure_masks.get(sid)
            break
    
    if reference_mask is None:
        raise ValueError(f"Reference structure '{reference_structure_name}' not found")
    
    print(f"Using '{structures[reference_sid]['name']}' as reference structure")
    print(f"Reference structure shape: {reference_mask.shape}")
    print(f"SUV volume shape: {suv_volume.shape}")
    
    # Get the CT shape that we'll need for the final structures
    ct_shape = None
    if ct_datasets and len(ct_datasets) > 0:
        ct_shape = (len(ct_datasets), ct_datasets[0].Rows, ct_datasets[0].Columns)
        print(f"Target CT shape for RT struct: {ct_shape}")
    
    # If SUV volume and reference mask shapes don't match, we need to handle this
    working_suv_volume = suv_volume
    working_reference_mask = reference_mask
    
    if suv_volume.shape != reference_mask.shape:
        print(f"⚠️ Warning: SUV volume shape {suv_volume.shape} != reference mask shape {reference_mask.shape}")
        # For now, we'll work with the reference mask shape and assume SUV volume needs to be resampled
        # This is a common scenario when PET and CT have different resolutions
        if ct_shape and reference_mask.shape == ct_shape:
            # Reference mask already matches CT, so resample SUV to match
            print("Resampling SUV volume to match reference structure...")
            working_suv_volume = resample_mask_to_ct_shape(suv_volume.astype(np.float32), reference_mask.shape)
        else:
            print("Working with mismatched shapes - results may not be optimal")
    
    new_structures = {}
    
    # 1. Create activity-based threshold structures (5000 Bq·mL steps)
    print("\n=== Creating Activity-Based Threshold Structures ===")
    
    # Get max activity from the working SUV volume
    roi_suv_values = working_suv_volume[working_reference_mask]
    max_activity = np.max(roi_suv_values) if roi_suv_values.size > 0 else 0
    print(f"Max activity in ROI: {max_activity:.3f} Bq·mL")
    
    # Create activity thresholds in 5000 Bq·mL steps (skip 0-5000, then 5000-10000, 10000-15000, etc)
    activity_thresholds_bqml = []
    current_activity = 5000  # Start at 5000 Bq·mL
    while current_activity <= max_activity:
        activity_thresholds_bqml.append(current_activity)
        current_activity += 5000
    
    if not activity_thresholds_bqml:
        activity_thresholds_bqml = [5000]  # Fallback: at least one threshold
    
    print(f"Activity thresholds (Bq·mL): {activity_thresholds_bqml}")
    
    # Create structures for each activity threshold
    activity_colors = {
        5000: [255, 0, 0],      # Red
        10000: [255, 128, 0],   # Orange
        15000: [255, 255, 0],   # Yellow
        20000: [128, 255, 0],   # Light green
        25000: [0, 255, 0],     # Green
        30000: [0, 255, 128],   # Cyan-green
        35000: [0, 255, 255],   # Cyan
    }
    
    threshold_structures = {}
    for threshold_activity in activity_thresholds_bqml:
        # Create mask for voxels >= this activity threshold
        if np.any(working_reference_mask):
            mask = working_reference_mask & (working_suv_volume >= threshold_activity)
            mask = find_largest_connected_component(mask)
            
            volume_ml = np.sum(mask) * voxel_volume_ml
            if volume_ml >= 0.1 and np.any(mask):
                roi_suv = working_suv_volume[mask]
                stats = {
                    'volume_ml': volume_ml,
                    'voxel_count': np.sum(mask),
                    'suv_mean': np.mean(roi_suv),
                    'suv_max': np.max(roi_suv),
                    'suv_min': np.min(roi_suv),
                    'threshold_activity_bqml': threshold_activity
                }
                threshold_structures[threshold_activity] = {'mask': mask, 'stats': stats}
                print(f"Created {threshold_activity}Bq·mL structure: {volume_ml:.3f} mL")
    
    for threshold_activity, data in threshold_structures.items():
        struct_name = f"{reference_structure_name}_{threshold_activity}BqmL"
        color = activity_colors.get(threshold_activity, [128, 128, 128])  # Gray fallback
        new_structures[struct_name] = {
            'mask': data['mask'],
            'color': color,
            'stats': data['stats']
        }
    
    # 2. Create gradient/edge structure
    print("\n=== Creating Edge Structure ===")
    edge_mask, edge_stats = create_gradient_structure(
        working_suv_volume, working_reference_mask,
        erosion_mm=2.0,
        voxel_spacing_mm=voxel_spacing_mm,
        voxel_volume_ml=voxel_volume_ml
    )
    
    if np.any(edge_mask):
        new_structures[f"{reference_structure_name}_Edge"] = {
            'mask': edge_mask,
            'color': [0, 255, 0],  # Green
            'stats': edge_stats
        }
    
    # 3. Create core structure (highly eroded)
    print("\n=== Creating Core Structure ===")
    
    # For core, we want the eroded region, not the rim
    structure_element = ndimage.generate_binary_structure(3, 1)
    erosion_voxels = [max(1, int(5.0 / spacing)) for spacing in voxel_spacing_mm]
    core_mask = working_reference_mask.copy()
    
    # Apply erosion
    for erosion in erosion_voxels:
        if erosion > 0:
            core_mask = ndimage.binary_erosion(core_mask, structure_element, iterations=erosion)
    
    if np.any(core_mask):
        core_volume_ml = np.sum(core_mask) * voxel_volume_ml
        if core_volume_ml >= 0.1:  # Minimum volume check
            core_suv_values = working_suv_volume[core_mask]
            core_stats = {
                'volume_ml': core_volume_ml,
                'voxel_count': np.sum(core_mask),
                'suv_mean': np.mean(core_suv_values),
                'suv_max': np.max(core_suv_values),
                'suv_min': np.min(core_suv_values)
            }
            
            new_structures[f"{reference_structure_name}_Core"] = {
                'mask': core_mask,
                'color': [0, 0, 255],  # Blue
                'stats': core_stats
            }
            print(f"Created core structure: {core_volume_ml:.3f} mL")
        else:
            print(f"Core structure too small ({core_volume_ml:.3f} mL), skipping")
    else:
        print("No core structure created (empty after erosion)")
    
    # Print summary of what was created
    print(f"\n=== Summary ===")
    print(f"Generated {len(new_structures)} structures from '{reference_structure_name}':")
    for name in new_structures.keys():
        print(f"  - {name}")
    
    return new_structures

def print_structure_summary(new_structures):
    """Print a summary of generated structures."""
    print("\n" + "="*60)
    print("GENERATED STRUCTURES SUMMARY")
    print("="*60)
    
    for struct_name, data in new_structures.items():
        stats = data['stats']
        color = data['color']
        
        print(f"\n{struct_name}:")
        print(f"  Volume: {stats.get('volume_ml', 0):.3f} mL")
        print(f"  Voxels: {stats.get('voxel_count', 0)}")
        print(f"  SUV mean: {stats.get('suv_mean', 0):.3f}")
        print(f"  SUV max: {stats.get('suv_max', 0):.3f}")
        print(f"  Color: RGB{tuple(color)}")
        
        if 'threshold_percent' in stats:
            print(f"  Threshold: {stats['threshold_percent']}% of max ({stats['threshold_suv']:.3f})")
    
    print(f"\nTotal structures generated: {len(new_structures)}")

def add_structure_generation_to_pvh(resampled_pet, structure_masks, structures, 
                                    ct_datasets, reference_structure_name,
                                    voxel_volume_ml=1.0, voxel_spacing_mm=[1.0, 1.0, 1.0], secondPatient=False, write_to_dicom=True):
    """
    Fixed version with better error handling and debugging
    
    Parameters:
    - write_to_dicom: If False, skip writing to DICOM file (useful when generating structures in a loop)
    """
    try:
        print("\n" + "="*60)
        print("GENERATING PET-BASED STRUCTURES")
        print("="*60)
        
        # Generate new structures
        new_structures = generate_pet_based_structures(
            resampled_pet, structure_masks, structures,
            reference_structure_name, ct_datasets,
            voxel_volume_ml, voxel_spacing_mm
        )
        
        if not new_structures:
            print("No structures were generated.")
            return None
        
        # Print summary
        print_structure_summary(new_structures)
        
        # Create output directory
        if secondPatient is False:
            output_dir = "generated_data/RS"
        else:
            output_dir = "generated_data2/RS"
        os.makedirs(output_dir, exist_ok=True)
        
        # Export structure statistics to CSV first (this always works)
        stats_path = os.path.join(output_dir, "Generated_Structure_Statistics.csv")
        export_structure_stats_csv(new_structures, stats_path)
        print(f"📊 Structure statistics exported to: {stats_path}")
        
        # Only attempt RT Structure Set creation if we have CT datasets AND write_to_dicom is True
        if write_to_dicom and ct_datasets and len(ct_datasets) > 0:
            try:
                # Write to RT Structure Set
                output_rs_path = os.path.join(output_dir, "Generated_PET_Structures.dcm")
                
                print(f"\nWriting structures to RT Structure Set...")
                write_structures_to_rt(
                    ct_datasets, new_structures, 
                    output_rs_path
                )
                
                print(f"✅ Generated RT Structure Set saved to: {output_rs_path}")
                
                # Now try to launch viewer with the generated structures
                try:
                    print("\n🔍 Preparing interactive viewer...")
                    
                    # Option 1: Use the new_structures directly (recommended)
                    print("Using directly generated structures for viewer...")
                    
                    # Convert new_structures to the format expected by viewer
                    viewer_masks = {}
                    viewer_structure_info = {}
                    
                    for i, (struct_name, struct_data) in enumerate(new_structures.items()):
                        viewer_masks[i] = struct_data['mask']
                        viewer_structure_info[i] = {"name": struct_name}
                        
                        # Debug: Check mask validity
                        mask = struct_data['mask']
                        print(f"Structure {i} ({struct_name}):")
                        print(f"  Shape: {mask.shape}")
                        print(f"  Voxels: {np.sum(mask)}")
                        print(f"  Shape matches SUV: {mask.shape == resampled_pet.shape}")
                        
                        if mask.shape != resampled_pet.shape:
                            print(f"  ⚠️ FIXING SHAPE MISMATCH")
                            # Try to fix shape mismatch
                            if len(mask.shape) == 3 and len(resampled_pet.shape) == 3:
                                try:
                                    fixed_mask = resample_mask_to_ct_shape(mask, resampled_pet.shape)
                                    viewer_masks[i] = fixed_mask
                                    print(f"  ✅ Fixed to shape: {fixed_mask.shape}")
                                except Exception as fix_error:
                                    print(f"  ❌ Could not fix shape: {fix_error}")
                                    # Remove this structure from viewer
                                    del viewer_masks[i]
                                    del viewer_structure_info[i]
                                    continue
                    
                    if not viewer_masks:
                        print("❌ No valid masks for viewer")
                        return new_structures
                    
                    # Launch viewer with directly generated structures
                    print(f"\n🧠 Launching viewer with {len(viewer_masks)} structures...")
                    
                    # Add debug info about SUV volume
                    print(f"SUV Volume for viewer:")
                    print(f"  Shape: {resampled_pet.shape}")
                    print(f"  Range: {np.min(resampled_pet):.3f} to {np.max(resampled_pet):.3f}")
                    print(f"  Non-zero voxels: {np.count_nonzero(resampled_pet)}")
                    
                    # Import the viewer
                    from processing.pvh import interactive_structure_viewer
                    
                    # Launch the viewer
                    interactive_structure_viewer(resampled_pet, viewer_masks, viewer_structure_info, secondPatient)
                    
                except Exception as viewer_error:
                    print(f"⚠️ Direct viewer launch failed: {viewer_error}")
                    print("Attempting alternative approach...")
                    
                    # Option 2: Try loading from the saved RT struct file
                    try:
                        import pydicom
                        from helper.binary_masks import create_structure_masks_with_registration
                        from helper.resample import Volume
                        
                        print("Loading generated RT Structure Set for viewer...")
                        
                        # Reload the generated RS file
                        generated_rs = pydicom.dcmread(output_rs_path)
                        
                        # Convert StructureSetROISequence to structure info dict
                        generated_structure_info = {
                            i: {"name": roi.ROIName}
                            for i, roi in enumerate(generated_rs.StructureSetROISequence)
                        }
                        
                        # Re-create masks using volume manager
                        volume_manager = Volume()
                        ct_volume, saved_cts = volume_manager.create_ct_volume(ct_datasets)
                        
                        generated_masks = create_structure_masks_with_registration(
                            structures=generated_structure_info,
                            rs_dataset=generated_rs,
                            volume_shape=resampled_pet.shape,  
                            reference_ds=saved_cts[0],
                            volume_instance=volume_manager,
                            pet_datasets=None 
                        )
                        
                        # Check if masks were created successfully
                        if generated_masks and any(mask is not None for mask in generated_masks.values()):
                            print("✅ Successfully recreated masks from RT struct")
                            
                            # Launch viewer
                            interactive_structure_viewer(resampled_pet, generated_masks, generated_structure_info, secondPatient)
                        else:
                            print("❌ Failed to recreate valid masks from RT struct")
                            
                    except Exception as reload_error:
                        print(f"❌ RT struct reload approach also failed: {reload_error}")
                        print("Generated structures are available for analysis, but viewer cannot be launched")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"⚠️ Warning: Could not create RT Structure Set: {e}")
                print("Generated structures are still available for analysis")
        else:
            print("⚠️ No CT datasets available - skipping RT Structure Set creation")
            print("Generated structures are available for analysis only")
        
        return new_structures
        
    except Exception as e:
        print(f"❌ Error generating PET-based structures: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_structure_stats_csv(new_structures, output_path):
    """Export generated structure statistics to CSV file."""

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = [
            "Structure_Name", "Volume_mL", "Voxel_Count", 
            "SUV_Mean", "SUV_Max", "SUV_Min", "Color_RGB",
            "Structure_Type", "Additional_Info"
        ]
        writer.writerow(header)
        
        # Write data for each structure
        for struct_name, struct_data in new_structures.items():
            stats = struct_data['stats']
            color = struct_data['color']
            
            # Determine structure type
            if 'bqml' in struct_name.lower():
                struct_type = "Activity_Threshold"
                additional_info = f"{stats.get('threshold_activity_bqml', 'N/A')} Bq·mL"
            elif 'edge' in struct_name.lower():
                struct_type = "Edge_Region"
                additional_info = f"Erosion: {stats.get('erosion_mm', 'N/A')}mm"
            elif 'core' in struct_name.lower():
                struct_type = "Core_Region"
                additional_info = "Eroded core region"
            else:
                struct_type = "Custom"
                additional_info = "Generated structure"
            
            row = [
                struct_name,
                f"{stats.get('volume_ml', 0):.3f}",
                stats.get('voxel_count', 0),
                f"{stats.get('suv_mean', 0):.3f}",
                f"{stats.get('suv_max', 0):.3f}",
                f"{stats.get('suv_min', 0):.3f}",
                f"({color[0]},{color[1]},{color[2]})",
                struct_type,
                additional_info
            ]
            writer.writerow(row)
