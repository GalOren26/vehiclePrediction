import tkinter as tk

def on_button_click():
    entered_text = text_entry.get()
    label.config(text=f"You entered: {entered_text}")

# Create the main window
root = tk.Tk()
root.title("Simple GUI")

# Create a label
label = tk.Label(root, text="Enter something:")
label.pack(padx=20, pady=10)

# Create a text entry box
text_entry = tk.Entry(root, width=20)
text_entry.pack(padx=20, pady=10)

# Create a button
button = tk.Button(root, text="Submit", command=on_button_click)
button.pack(padx=20, pady=10)

# Start the event loop
root.mainloop()