import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import subprocess
import sys
import threading


class StressAnalysisApp:
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.df = None
        self.file_path = None
        self._configure_window()
        self._create_styles()
        self._create_widgets()

    def _configure_window(self):
        """Configure the main window properties."""
        self.root.title("📊 XGBoost Stress Analysis Launcher")
        self.root.geometry("900x600")
        self.root.minsize(600, 400)

    def _create_styles(self):
        """Create and configure ttk styles for a modern look."""
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 12), padding=10)
        style.configure("Info.TLabel", font=("Segoe UI", 12))
        style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"))
        style.configure("Treeview", rowheight=25, font=("Segoe UI", 10))
        style.configure("Success.TButton", background="#007bff", foreground="white")
        style.map("Success.TButton", background=[("active", "#0056b3")])

    def _create_widgets(self):
        """Create and arrange all the widgets in the window."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        self.upload_btn = ttk.Button(
            top_frame, text="📂 1. Upload CSV", command=self.upload_csv
        )
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.analyze_btn = ttk.Button(
            top_frame,
            text="🚀 2. Run XGBoost Analysis",
            command=self.run_analysis_threaded,
            state=tk.DISABLED,
            style="Success.TButton",
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.info_label = ttk.Label(
            top_frame,
            text="Please upload a survey CSV file to begin.",
            style="Info.TLabel",
        )
        self.info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_frame, show="headings")
        v_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        h_scroll = ttk.Scrollbar(
            tree_frame, orient="horizontal", command=self.tree.xview
        )
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        v_scroll.pack(side=tk.RIGHT, fill="y")
        h_scroll.pack(side=tk.BOTTOM, fill="x")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def upload_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select a Survey CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not file_path:
            return
        self.file_path = file_path
        try:
            self.df = pd.read_csv(self.file_path)
            self._display_dataframe_preview()
            file_name = os.path.basename(self.file_path)
            self.info_label.config(text=f"✅ Loaded: {file_name}")
            self.analyze_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read the file:\n{e}")
            self.analyze_btn.config(state=tk.DISABLED)

    def _display_dataframe_preview(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["column"] = list(self.df.columns)
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=120)
        for _, row in self.df.head(100).iterrows():
            self.tree.insert("", "end", values=list(row))

    def run_analysis_threaded(self):
        """Runs the analysis in a separate thread to keep the UI from freezing."""
        self.analyze_btn.config(state=tk.DISABLED, text="⏳ Optimizing XGBoost...")
        self.root.update_idletasks()

        # Run the analysis in a background thread
        thread = threading.Thread(target=self.run_analysis_subprocess)
        thread.daemon = (
            True  # Allows the main app to exit even if the thread is running
        )
        thread.start()

    def run_analysis_subprocess(self):
        """Calls the stress.py script and handles its output."""
        if not self.file_path:
            messagebox.showwarning("Warning", "No file path is available.")
            self.analyze_btn.config(state=tk.NORMAL, text="🚀 2. Run XGBoost Analysis")
            return

        command = [sys.executable, "stress.py", self.file_path]
        try:
            print(f"▶️  Running analysis script...")
            # **FIX**: Run the subprocess and allow its output to stream to the console.
            # This lets you see the GridSearch progress.
            # We also capture stderr to display it in case of an error.
            process = subprocess.run(
                command, check=True, capture_output=True, text=True, encoding="utf-8"
            )
            print(process.stdout)  # Print any output from the script upon success
            print("✅ Analysis complete.")

        except FileNotFoundError:
            messagebox.showerror(
                "Error",
                "Could not find 'stress.py'. Make sure it is in the same folder.",
            )
        except subprocess.CalledProcessError as e:
            # This is the key fix: Display the actual error from the script
            messagebox.showerror(
                "Analysis Script Error",
                f"The analysis script failed to run.\n\nERROR:\n{e.stderr}",
            )
        except Exception as e:
            messagebox.showerror("An Unexpected Error Occurred", str(e))
        finally:
            # Always reset the button text and state
            self.analyze_btn.config(state=tk.NORMAL, text="🚀 2. Run XGBoost Analysis")


if __name__ == "__main__":
    root = tk.Tk()
    app = StressAnalysisApp(root)
    root.mainloop()
