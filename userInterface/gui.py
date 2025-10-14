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

def select_patient_gui(patients_dict):
    """
    GUI to select a patient from discovered patients.
    """
    root = tk.Tk()
    root.title("Select Patient")
    root.geometry("500x400")
    
    selected = {'patient': None}
    
    tk.Label(root, text="Select Patient to Analyze:", font=('Arial', 12, 'bold')).pack(pady=10)
    
    listbox = tk.Listbox(root, height=15, font=('Arial', 10))
    listbox.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
    
    for patient_name, patient_info in patients_dict.items():
        num_timepoints = len(patient_info['timepoints'])
        display_text = f"{patient_name} ({num_timepoints} timepoint{'s' if num_timepoints > 1 else ''})"
        listbox.insert(tk.END, display_text)
    
    def on_submit():
        selection = listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a patient")
            return
        
        patient_names = list(patients_dict.keys())
        selected['patient'] = patient_names[selection[0]]
        root.quit()
        root.destroy()
    
    tk.Button(root, text="Select", command=on_submit,
              bg='green', fg='white', font=('Arial', 11, 'bold')).pack(pady=20)
    
    root.mainloop()
    
    try:
        root.destroy()
    except:
        pass
    
    return selected['patient']
