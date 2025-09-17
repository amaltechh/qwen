import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import subprocess
import sys
import threading
from PIL import Image, ImageTk
import sv_ttk

class StressAnalysisApp:
    """
    Initializes the StressAnalysisApp.
    """
    def __init__(self, root):
        self.root = root
        self.df = None
        self.file_path = None

        # --- UI Configuration ---
        self.ICON_SIZE = (24, 24)
        self.FONT_FAMILY = "Roboto"
        
        # --- Asset Loading ---
        self.upload_icon = self._load_icon("icons/upload.png")
        self.analyze_icon = self._load_icon("icons/run.png")
        
        # --- Window and Style Setup ---
        self._configure_window()
        self._create_styles()
        
        # --- Widget Creation ---
        self._create_widgets()
        self._fade_in_window()

    def _load_icon(self, path):
        """
        Loads and resizes an icon.
        
        Args:
            path (str): The path to the icon file.
            
        Returns:
            ImageTk.PhotoImage: The loaded and resized icon.
        """
        try:
            img = Image.open(path).resize(self.ICON_SIZE, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading icon {path}: {e}")
            return None

    def _configure_window(self):
        """
        Configures the main window properties.
        """
        self.root.title("NeuroGlimpse - XGBoost Stress Analyzer")
        self.root.geometry("1300x800")
        self.root.minsize(1000, 650)
        sv_ttk.set_theme("dark")
        self.root.attributes("-alpha", 0.0)

    def _create_styles(self):
        """
        Creates and configures ttk styles for a modern look.
        """
        style = ttk.Style(self.root)
        
        style.configure("Accent.TButton", font=(self.FONT_FAMILY, 13, "bold"), padding=(18, 15))
        style.configure("TButton", font=(self.FONT_FAMILY, 12), padding=(12, 12))
        style.configure("Header.TLabel", font=(self.FONT_FAMILY, 22, "bold"), padding=(0, 18, 0, 8))
        style.configure("Status.TLabel", font=(self.FONT_FAMILY, 11))
        style.configure("Placeholder.TLabel", font=(self.FONT_FAMILY, 16, "italic"))
        style.configure("Treeview.Heading", font=(self.FONT_FAMILY, 12, "bold"))
        style.configure("Treeview", rowheight=32, font=(self.FONT_FAMILY, 11))

    def _create_widgets(self):
        """
        Creates and arranges all the widgets in the window.
        """
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.sidebar_frame = ttk.Frame(paned_window, width=300)
        self._create_sidebar(self.sidebar_frame)
        paned_window.add(self.sidebar_frame, weight=1)

        self.main_content_frame = ttk.Frame(paned_window)
        self._create_main_content(self.main_content_frame)
        paned_window.add(self.main_content_frame, weight=4)

        self._create_statusbar()
        
    def _create_sidebar(self, parent):
        """
        Creates the sidebar with controls.
        
        Args:
            parent (tk.Widget): The parent widget.
        """
        parent.pack_propagate(False)
        
        header = ttk.Label(parent, text="CONTROLS", style="Header.TLabel")
        header.pack(pady=(20, 30), padx=25, anchor="w")

        self.upload_btn = ttk.Button(
            parent, text="Upload Survey CSV", image=self.upload_icon,
            compound="left", command=self.upload_csv
        )
        self.upload_btn.pack(fill=tk.X, padx=25, pady=8)
        
        self.analyze_btn = ttk.Button(
            parent, text="Run XGBoost Analysis", image=self.analyze_icon,
            compound="left", command=self.run_analysis_threaded,
            state=tk.DISABLED, style="Accent.TButton"
        )
        self.analyze_btn.pack(fill=tk.X, padx=25, pady=(12, 25))
        
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=25, pady=20)
        
        self.progress_label = ttk.Label(parent, text="Analysis Progress", font=(self.FONT_FAMILY, 11, "italic"))
        self.progress_label.pack(padx=25, anchor="w")
        
        self.progress_bar = ttk.Progressbar(parent, orient="horizontal", mode="indeterminate")
        self.progress_bar.pack(fill=tk.X, padx=25, pady=(8, 18))

    def _create_main_content(self, parent):
        """
        Creates the data preview area.
        
        Args:
            parent (tk.Widget): The parent widget.
        """
        self.placeholder_label = ttk.Label(parent, text="Upload a CSV file to see a data preview...", style="Placeholder.TLabel")
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.tree = ttk.Treeview(parent, show="headings")
        v_scroll = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        h_scroll = ttk.Scrollbar(parent, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.tree_v_scroll = v_scroll
        self.tree_h_scroll = h_scroll

    def _create_statusbar(self):
        """
        Creates the bottom status bar.
        """
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(0, 15))
        self.status_label = ttk.Label(status_frame, text="Welcome! Please upload a survey CSV.", anchor="w", style="Status.TLabel")
        self.status_label.pack(fill=tk.X)

    def upload_csv(self):
        """
        Handles the CSV file upload and display with a progress animation.
        """
        file_path = filedialog.askopenfilename(
            title="Select a Survey CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not file_path:
            return

        self.file_path = file_path
        
        # --- Animate file reading ---
        self.progress_bar.config(mode="determinate")
        self.status_label.config(text=f"Reading {os.path.basename(file_path)}...")
        
        def read_in_chunks():
            try:
                chunk_list = []
                total_size = os.path.getsize(self.file_path)
                bytes_read = 0
                
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    for chunk in pd.read_csv(f, chunksize=1000):
                        chunk_list.append(chunk)
                        # Estimate progress; not perfect but gives good feedback
                        bytes_read = min(total_size, bytes_read + chunk.memory_usage(deep=True).sum())
                        progress = (bytes_read / total_size) * 100
                        self.progress_bar['value'] = progress
                        self.root.update_idletasks()
                
                self.df = pd.concat(chunk_list, ignore_index=True)
                self._display_dataframe_preview()
                file_name = os.path.basename(self.file_path)
                self.status_label.config(text=f"✅ Loaded: {file_name} | {self.df.shape[0]} rows, {self.df.shape[1]} columns")
                self.analyze_btn.config(state=tk.NORMAL)
                self.placeholder_label.place_forget()
            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read the file:\n{e}")
                self.status_label.config(text="❌ Error loading file.")
                self.analyze_btn.config(state=tk.DISABLED)
            finally:
                self.progress_bar['value'] = 0
                self.progress_bar.config(mode="indeterminate")

        threading.Thread(target=read_in_chunks, daemon=True).start()


    def _display_dataframe_preview(self):
        """
        Clears and populates the Treeview with DataFrame content.
        """
        self.tree.delete(*self.tree.get_children())
        self.tree["column"] = list(self.df.columns)
        
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="w", width=170, stretch=tk.NO)
            
        for _, row in self.df.head(200).iterrows():
            self.tree.insert("", "end", values=list(row))
        
        self.tree_v_scroll.pack(side=tk.RIGHT, fill="y")
        self.tree_h_scroll.pack(side=tk.BOTTOM, fill="x")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def run_analysis_threaded(self):
        """
        Runs the analysis in a separate thread to keep the UI responsive.
        """
        self.analyze_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏳ Optimizing XGBoost... Please wait.")
        self.progress_bar.start(10)
        
        thread = threading.Thread(target=self.run_analysis_subprocess, daemon=True)
        thread.start()

    def run_analysis_subprocess(self):
        """
        Calls the stress.py script and handles its output.
        """
        if not self.file_path:
            messagebox.showwarning("Warning", "No file path is available.")
            self._reset_ui_state()
            return

        command = [sys.executable, "stress.py", self.file_path]
        try:
            print(f"▶️ Running analysis script...")
            process = subprocess.run(
                command, check=True, capture_output=True, text=True, encoding="utf-8"
            )
            print(process.stdout)
            messagebox.showinfo("Success", "Analysis complete! Results have been generated in the script's folder.")
            self.status_label.config(text="✅ Analysis complete successfully.")
        except FileNotFoundError:
            messagebox.showerror("Error", "Could not find 'stress.py'. Make sure it is in the same folder.")
            self.status_label.config(text="❌ Error: 'stress.py' not found.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror(
                "Analysis Script Error",
                f"The analysis script failed to run.\n\nERROR:\n{e.stderr}",
            )
            self.status_label.config(text="❌ Analysis script failed with an error.")
        except Exception as e:
            messagebox.showerror("An Unexpected Error Occurred", str(e))
            self.status_label.config(text=f"❌ An unexpected error occurred: {e}")
        finally:
            self.root.after(100, self._reset_ui_state)

    def _reset_ui_state(self):
        """
        Resets the UI controls to their default state after analysis.
        """
        self.analyze_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.progress_bar.stop()

    def _fade_in_window(self):
        """
        Fades the window in from transparent to opaque.
        """
        alpha = self.root.attributes("-alpha")
        if alpha < 1:
            alpha += 0.05
            self.root.attributes("-alpha", alpha)
            self.root.after(12, self._fade_in_window)

if __name__ == "__main__":
    root = tk.Tk()
    app = StressAnalysisApp(root)
    root.mainloop()