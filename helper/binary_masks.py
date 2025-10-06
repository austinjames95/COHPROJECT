from dicompylercore import dicomparser
import numpy as np
from skimage.draw import polygon as draw_polygon
from shapely.geometry import Polygon

def create_structure_masks(structures, rs_dataset, volume_shape, reference_ds,
                           registration_transform=None, pet_datasets=None, ct_datasets=None):
    rt = dicomparser.DicomParser(rs_dataset)

    origin = np.array(reference_ds.ImagePositionPatient)
    spacing = list(map(float, reference_ds.PixelSpacing))
    thickness = float(getattr(reference_ds, 'SliceThickness', 1.0))
    spacing.append(thickness)

    # Determine z-slice positions based on modality - FIXED
    if reference_ds.Modality == 'CT':
        if ct_datasets:
            z_positions = sorted([float(ds.ImagePositionPatient[2]) for ds in ct_datasets])
        else:
            # Fallback to single reference dataset if ct_datasets not provided
            z_positions = [float(reference_ds.ImagePositionPatient[2])]
    elif reference_ds.Modality == 'PT' and pet_datasets:
        z_positions = sorted([float(ds.ImagePositionPatient[2]) for ds in pet_datasets])
    else:
        raise RuntimeError("Cannot determine z-positions for mask creation — missing CT or PET dataset.")

    if not z_positions:
        raise RuntimeError("No valid image slices to match against RTSTRUCT z-coordinates.")

    print(f"Reference modality: {reference_ds.Modality}")
    print(f"Number of z-positions: {len(z_positions)}")
    print(f"Z-position range: {min(z_positions):.2f} to {max(z_positions):.2f}")
    print(f"Volume shape: {volume_shape}")

    masks = {}

    for sid, struct in structures.items():
        coords = rt.GetStructureCoordinates(sid)
        if not coords:
            continue

        mask = np.zeros(volume_shape, dtype=bool)
        processed_slices = 0

        for z_str, contours in coords.items():
            z = float(z_str)
            
            # Find closest z-slice - improved matching
            z_diffs = np.abs(np.array(z_positions) - z)
            k = np.argmin(z_diffs)
            
            # Add tolerance check to avoid matching distant slices
            min_diff = z_diffs[k]
            if min_diff > thickness * 2:  # Allow up to 2x slice thickness tolerance
                print(f"Warning: Structure z={z:.2f} is {min_diff:.2f}mm from nearest slice at z={z_positions[k]:.2f}")
                continue
            
            # Ensure k is within volume bounds
            if k >= volume_shape[0]:
                print(f"Warning: Slice index {k} exceeds volume depth {volume_shape[0]}")
                continue

            canvas = np.zeros((volume_shape[1], volume_shape[2]), dtype=np.uint8)
            polygon_list = []

            for contour in contours:
                pts = contour['data']
                if len(pts) < 3:
                    continue

                pixel_pts = []
                for pt in pts:
                    x_mm, y_mm, z_mm = pt
                    
                    # Apply registration transform if provided
                    if registration_transform:
                        try:
                            x_mm, y_mm, z_mm = registration_transform.TransformPoint([x_mm, y_mm, z_mm])
                        except Exception as e:
                            print(f"Warning: Registration transform failed for point {pt}: {e}")
                    
                    # Convert to pixel coordinates
                    x_pix = (x_mm - origin[0]) / spacing[0]
                    y_pix = (y_mm - origin[1]) / spacing[1]
                    
                    # Bounds checking
                    if (0 <= x_pix < volume_shape[2] and 0 <= y_pix < volume_shape[1]):
                        pixel_pts.append([x_pix, y_pix])

                if len(pixel_pts) >= 3:  # Need at least 3 points for a valid polygon
                    try:
                        poly = Polygon(pixel_pts)
                        if poly.is_valid:
                            polygon_list.append((poly, pixel_pts))
                    except Exception as e:
                        print(f"Warning: Invalid polygon for structure {struct.get('name', sid)}: {e}")

            # Separate outer polygons from holes
            outer_polys, hole_polys = [], []
            for i, (poly_i, pix_i) in enumerate(polygon_list):
                is_hole = any(j != i and poly_j.contains(poly_i) for j, (poly_j, _) in enumerate(polygon_list))
                (hole_polys if is_hole else outer_polys).append(pix_i)

            def fill(canvas, points):
                if len(points) < 3: 
                    return
                try:
                    points_array = np.array(points)
                    rr, cc = draw_polygon(points_array[:, 1], points_array[:, 0], canvas.shape)
                    # Bounds checking for polygon fill
                    valid_indices = (rr >= 0) & (rr < canvas.shape[0]) & (cc >= 0) & (cc < canvas.shape[1])
                    canvas[rr[valid_indices], cc[valid_indices]] = 1
                except Exception as e:
                    print(f"Warning: Failed to fill polygon: {e}")

            def subtract(canvas, points):
                if len(points) < 3: 
                    return
                try:
                    points_array = np.array(points)
                    rr, cc = draw_polygon(points_array[:, 1], points_array[:, 0], canvas.shape)
                    # Bounds checking for polygon subtraction
                    valid_indices = (rr >= 0) & (rr < canvas.shape[0]) & (cc >= 0) & (cc < canvas.shape[1])
                    canvas[rr[valid_indices], cc[valid_indices]] = 0
                except Exception as e:
                    print(f"Warning: Failed to subtract polygon: {e}")

            # Fill outer polygons and subtract holes
            for poly in outer_polys: 
                fill(canvas, poly)
            for poly in hole_polys: 
                subtract(canvas, poly)

            # Apply to mask
            mask[k] |= (canvas > 0)
            processed_slices += 1

        masks[sid] = mask
        print(f"Structure '{struct.get('name', sid)}': processed {processed_slices} slices, mask volume: {np.sum(mask)} voxels")

    print(f"Created masks for {len(masks)} structures")
    return masks

def create_structure_masks_with_registration(structures, rs_dataset, volume_shape,
                                             reference_ds, volume_instance=None,
                                             structure_reg_transform=None,
                                             pet_datasets=None, ct_datasets=None):
    registration_transform = structure_reg_transform or (
        volume_instance.get_registration_transform() if volume_instance else None
    )

    return create_structure_masks(structures, rs_dataset, volume_shape, reference_ds,
                                  registration_transform=registration_transform,
                                  pet_datasets=pet_datasets, ct_datasets=ct_datasets)
