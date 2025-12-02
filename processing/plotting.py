import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_pvh(pvh_table_absolute, pvh_table_relative, secondTimePoint):
    num_structures = len([k for k in pvh_table_absolute.keys() if k != 'BQML'])
    colors = plt.cm.tab10(np.linspace(0, 1, num_structures)) if num_structures <= 10 else plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    pet_values = pvh_table_absolute["BQML"]
    plt.figure(figsize=(12, 8))
    for i, (struct_name, volumes) in enumerate(pvh_table_relative.items()):
        if struct_name == "BQML":
            continue
        clean_name = struct_name.replace("_RelativeCumVolume", "").replace("_", " ")
        plt.plot(pvh_table_relative["BQML"], volumes, label=clean_name, linewidth=2, color=colors[i % len(colors)])

    plt.xlabel('PET Activity (Bq/mL)')
    plt.ylabel('Relative Volume (fraction)')
    plt.title('Cumulative PVH - Relative Volumes')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if secondTimePoint is False:
        os.makedirs("generated_data/PVH", exist_ok=True)
        plt.savefig("generated_data/PVH/PVH_Relative_Plot.png", dpi=300, bbox_inches='tight')
    else:
        os.makedirs("generated_data2/PVH", exist_ok=True)
        plt.savefig("generated_data2/PVH/PVH_Relative_Plot.png", dpi=300, bbox_inches='tight')

    print("Relative PVH plot saved.")


def plot_dvh(dvh_table_absolute, dvh_table_relative, dvh_out_dir):
    num_structures = len([k for k in dvh_table_absolute.keys() if k != 'GY'])
    if num_structures <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_structures))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    dose_values = dvh_table_absolute["GY"]

    structure_index = 0

    plt.figure(figsize=(12, 8))
    structure_index = 0
    for struct_name, volumes in dvh_table_relative.items():
        if struct_name == "GY":
            continue
        
        clean_name = struct_name.replace("_RelativeCumVolume", "").replace("_", " ")
        plt.plot(dvh_table_relative["GY"], volumes, 
                label=clean_name, 
                linewidth=2,
                color=colors[structure_index % len(colors)])
        structure_index += 1

    plt.xlabel('Dose (Gy)', fontsize=12)
    plt.ylabel('Relative Volume (fraction)', fontsize=12)
    plt.title('Cumulative DVH - Relative Volumes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    rel_plot_path = os.path.join(dvh_out_dir, "DVH_Relative_Plot.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
    print(f" Relative DVH plot saved to: {rel_plot_path}")
    
    print("\nDVH Summary Statistics:")
    print("="*60)
    
    for struct_name, volumes in dvh_table_absolute.items():
        if struct_name == "GY":
            continue
            
        clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
        max_volume = volumes[0] 

        rel_struct_name = struct_name.replace("_Volume_cm3", "")
        if rel_struct_name in dvh_table_relative:
            rel_volumes = dvh_table_relative[rel_struct_name]
            
            dose_95_vol = None
            dose_50_vol = None
            dose_5_vol = None
            
            for j, rel_vol in enumerate(rel_volumes):
                if dose_95_vol is None and rel_vol <= 0.95:
                    dose_95_vol = dose_values[j]
                if dose_50_vol is None and rel_vol <= 0.50:
                    dose_50_vol = dose_values[j]
                if dose_5_vol is None and rel_vol <= 0.05:
                    dose_5_vol = dose_values[j]
        
        print(f"\n{clean_name}:")
        print(f"  Total Volume: {max_volume:.2f} cm³")
        if 'dose_95_vol' in locals() and dose_95_vol is not None:
            print(f"  D95 (dose to 95% volume): {dose_95_vol:.2f} Gy")
        if 'dose_50_vol' in locals() and dose_50_vol is not None:
            print(f"  D50 (dose to 50% volume): {dose_50_vol:.2f} Gy")
        if 'dose_5_vol' in locals() and dose_5_vol is not None:
            print(f"  D5 (dose to 5% volume): {dose_5_vol:.2f} Gy")
    
    print("\n" + "="*60)
    print("\nProcessing completed successfully!")
