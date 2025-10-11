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

