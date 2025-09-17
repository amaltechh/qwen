# analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox, filedialog as fd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sys
import time
from datetime import datetime
import threading
import csv
import traceback
import itertools
import numpy as np
import os
import subprocess


# ----------------------- Utility -----------------------
def log_step(message, start_time=None):
    now = datetime.now().strftime("%H:%M:%S")
    if start_time is None:
        print(f"[{now}] {message}...")
        return time.time()
    else:
        print(f"[{now}] {message} (took {time.time() - start_time:.2f} sec)")
        return time.time() - start_time


# ----------------------- Data Loader -----------------------
def load_and_preprocess_slowly(file_path, update_callback):
    log_step("Starting slow data processing")
    with open(file_path, "r", encoding="utf-8") as f:
        total_rows = sum(1 for line in f) - 1

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        update_callback("headers", headers)

        likert_mapping = {
            "Strongly Disagree": 1,
            "Disagree": 2,
            "Neutral": 3,
            "Agree": 4,
            "Strongly Agree": 5,
        }
        likert_indices = [i for i, h in enumerate(headers) if "Q" in h]

        df_rows, numeric_rows, start_time = [], [], time.time()
        for i, row in enumerate(reader):
            df_rows.append(row)
            numeric_row = [
                likert_mapping.get(cell, 0)
                for idx, cell in enumerate(row)
                if idx in likert_indices
            ]
            numeric_rows.append(numeric_row)

            progress = ((i + 1) / total_rows) * 70
            elapsed_time = time.time() - start_time
            etr = (elapsed_time / (i + 1)) * (total_rows - (i + 1))
            update_callback(
                "row", (row, progress, etr, f"Reading row {i+1} of {total_rows}")
            )
            time.sleep(0.01)

    df = pd.DataFrame(df_rows, columns=headers)
    likert_column_names = [headers[i] for i in likert_indices]
    df_numeric = pd.DataFrame(numeric_rows, columns=likert_column_names)

    total_scores = df_numeric.sum(axis=1)
    stress_labels = ["No stress", "Low stress", "Medium stress", "High stress"]
    try:
        df["Stress_Level"] = pd.qcut(
            total_scores, q=4, labels=stress_labels, duplicates="drop"
        )
    except ValueError:
        df["Stress_Level"] = pd.cut(
            total_scores, bins=4, labels=stress_labels, include_lowest=True
        )

    log_step(
        f"Stress levels classified into {df['Stress_Level'].nunique()} adaptive categories."
    )
    return df, df_numeric


# ----------------------- Training -----------------------
def train_and_get_data(X, y_encoded, encoder, original_y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
    )

    t1 = log_step("Training XGBoost model")
    model.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False
    )
    training_time = log_step("Model training complete", t1)

    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred_test)

    all_class_indices = np.arange(len(encoder.classes_))
    report = classification_report(
        y_test,
        y_pred_test,
        labels=all_class_indices,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm_data = confusion_matrix(y_test, y_pred_test, labels=all_class_indices)

    raw_data_pack = {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "training_time": training_time,
        "report_dict": report,
        "cm_data": cm_data,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba_test,
        "importance": model.get_booster().get_score(importance_type="weight"),
        "learning_curves": model.evals_result(),
        "distribution": original_y.value_counts(),
        "data_profile": {"participants": X.shape[0], "questions": X.shape[1]},
    }
    return raw_data_pack


# ----------------------- Results Window -----------------------
class ResultsWindow:
    def __init__(self, master, data_pack, figures, full_df=None):
        self.master = master
        self.df = full_df  # store dataset with Stress_Level
        self.top = tk.Toplevel(master)
        self.top.title("📊 Professional Stress Analysis Report")
        self.top.geometry("1400x900")
        self.top.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style(self.top)
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Bold.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))

        paned_window = ttk.PanedWindow(self.top, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel
        left_frame = ttk.Frame(paned_window, width=450)
        paned_window.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Analysis Dashboard", style="Header.TLabel").pack(
            pady=(0, 15), anchor="w"
        )

        metrics_frame = ttk.LabelFrame(left_frame, text="Key Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Label(metrics_frame, text="Training Accuracy:", style="Bold.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(metrics_frame, text=f"{data_pack['train_acc'] * 100:.2f}%").grid(
            row=0, column=1, sticky="w", padx=5
        )
        ttk.Label(metrics_frame, text="Test Accuracy:", style="Bold.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(metrics_frame, text=f"{data_pack['test_acc'] * 100:.2f}%").grid(
            row=1, column=1, sticky="w", padx=5
        )
        ttk.Label(metrics_frame, text="Training Time:", style="Bold.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(metrics_frame, text=f"{data_pack['training_time']:.2f} sec").grid(
            row=2, column=1, sticky="w", padx=5
        )

        profile_frame = ttk.LabelFrame(left_frame, text="Data Profile", padding=10)
        profile_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Label(profile_frame, text="Participants:", style="Bold.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            profile_frame, text=f"{data_pack['data_profile']['participants']}"
        ).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(profile_frame, text="Questions Analyzed:", style="Bold.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(profile_frame, text=f"{data_pack['data_profile']['questions']}").grid(
            row=1, column=1, sticky="w", padx=5
        )

        # Button to open stress report
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(
            btn_frame,
            text="📥 View & Export Stress Levels",
            command=self.open_stress_report,
        ).pack(side=tk.RIGHT, padx=10)

        # Classification Report
        report_frame = ttk.LabelFrame(
            left_frame, text="Classification Report (Text)", padding=10
        )
        report_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        report_text = pd.DataFrame(data_pack["report_dict"]).T.to_string()
        report_box = scrolledtext.ScrolledText(
            report_frame, height=10, font=("Courier New", 9), relief="solid", bd=1
        )
        report_box.insert(tk.END, report_text)
        report_box.config(state=tk.DISABLED)
        report_box.pack(fill=tk.BOTH, expand=True)

        # Right panel
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=3)
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        fig_titles = [
            "Confusion Matrix",
            "Normalized CM",
            "Report Heatmap",
            "Feature Importance",
            "Learning Curve",
            "ROC Curves",
            "Precision-Recall",
            "Data Distribution",
        ]
        for title, figure in zip(fig_titles, figures):
            self.create_plot_tab(notebook, title, figure)

    def create_plot_tab(self, notebook, title, figure):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def open_stress_report(self):
        """Open a new window with full dataset + Stress_Level + export button."""
        if self.df is None:
            messagebox.showerror("Error", "No dataset available for report")
            return

        win = tk.Toplevel(self.top)
        win.title("🧠 Individual Stress Level Report")
        win.geometry("1000x600")

        tree = ttk.Treeview(win, show="headings")
        vsb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        # Setup headers
        tree["columns"] = list(self.df.columns)
        for col in self.df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")

        # Insert rows
        for _, row in self.df.iterrows():
            tree.insert("", "end", values=list(row))

        # Export button
        def export_data():
            file_path = fd.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")],
            )
            if not file_path:
                return
            try:
                if file_path.endswith(".csv"):
                    self.df.to_csv(file_path, index=False)
                else:
                    self.df.to_excel(file_path, index=False)
                messagebox.showinfo("✅ Export Successful", f"Saved to {file_path}")
            except Exception as e:
                messagebox.showerror("❌ Export Failed", str(e))

        ttk.Button(win, text="💾 Download Data", command=export_data).pack(pady=10)

    def on_close(self):
        print("INFO: Closing report and launching Stress Test Quiz...")
        try:
            subprocess.Popen([sys.executable, "test.py"])
        except FileNotFoundError:
            messagebox.showerror(
                "Error",
                "Could not find 'test.py'. Make sure it's saved in the same folder.",
            )
        except Exception as e:
            messagebox.showerror("Execution Error", f"Failed to launch test.py:\n{e}")
        self.master.destroy()


# ----------------------- Processing Window -----------------------
class ProcessingWindow:
    def __init__(self, master, file_path):
        self.master = master
        self.file_path = file_path
        self.df = None
        self.master.title("Stress Analysis 🧠")
        self.master.geometry("800x600")

        style = ttk.Style(self.master)
        style.theme_use("clam")

        top_frame = ttk.Frame(master, padding=(20, 10))
        top_frame.pack(fill=tk.X)
        self.status_label = ttk.Label(top_frame, text="Ready...", font=("Segoe UI", 11))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.etr_label = ttk.Label(top_frame, text="ETR: --", font=("Segoe UI", 11))
        self.etr_label.pack(side=tk.RIGHT)

        progress_frame = ttk.Frame(master, padding=(20, 0))
        progress_frame.pack(fill=tk.X)
        self.progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress.pack(fill=tk.X, expand=True)

        tree_frame = ttk.Frame(master, padding=(20, 10))
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_frame, show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)

        bottom_frame = ttk.Frame(master, padding=(20, 10))
        bottom_frame.pack(fill=tk.X)
        self.start_button = ttk.Button(
            bottom_frame,
            text="🚀 Start Professional Analysis",
            command=self.start_processing_thread,
        )
        self.start_button.pack()

    def _setup_treeview_headers(self, headers):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = headers
        [self.tree.heading(c, text=c) for c in headers]

    def _update_gui(self, data):
        row, p, etr, s = data
        self.status_label.config(text=s)
        self.etr_label.config(text=f"ETR: {etr:.1f}s")
        self.progress["value"] = p
        self.tree.insert("", "end", values=row)
        self.tree.yview_moveto(1.0)

    def start_processing_thread(self):
        self.start_button.config(state=tk.DISABLED, text="Processing...")
        self.thread = threading.Thread(target=self.run_pipeline, daemon=True)
        self.thread.start()

    def create_visuals(self, data_pack, encoder):
        log_step("Generating visuals on main thread")
        figures = []
        try:
            # Confusion Matrix
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                data_pack["cm_data"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                ax=ax_cm,
            )
            ax_cm.set_title("Confusion Matrix (Counts)")
            figures.append(fig_cm)

            # Normalized CM
            cm_norm = data_pack["cm_data"].astype("float") / data_pack["cm_data"].sum(
                axis=1, keepdims=True
            )
            cm_norm = np.nan_to_num(cm_norm)
            fig_norm, ax_norm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2%",
                cmap="Greens",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                ax=ax_norm,
            )
            ax_norm.set_title("Normalized Confusion Matrix (%)")
            figures.append(fig_norm)

            # Report Heatmap
            fig_rep, ax_rep = plt.subplots(figsize=(8, 5))
            sns.heatmap(
                pd.DataFrame(data_pack["report_dict"]).iloc[:-1, :].T,
                annot=True,
                cmap="viridis",
                ax=ax_rep,
            )
            ax_rep.set_title("Classification Report Heatmap")
            figures.append(fig_rep)

            # Feature Importance
            fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
            sorted_imp = sorted(data_pack["importance"].items(), key=lambda i: i[1])
            df_imp = pd.DataFrame(sorted_imp, columns=["F", "S"])
            ax_fi.barh(df_imp["F"], df_imp["S"])
            ax_fi.set_title("Feature Importance")
            plt.tight_layout()
            figures.append(fig_fi)

            # Learning Curve
            fig_lc, ax_lc = plt.subplots(figsize=(8, 5))
            res = data_pack["learning_curves"]
            ax_lc.plot(res["validation_0"]["mlogloss"], label="Train")
            ax_lc.plot(res["validation_1"]["mlogloss"], label="Test")
            ax_lc.legend()
            ax_lc.set_title("Learning Curve")
            figures.append(fig_lc)

            # ROC & PR curves
            if len(encoder.classes_) > 1 and len(np.unique(data_pack["y_test"])) > 1:
                lb = LabelBinarizer().fit(range(len(encoder.classes_)))
                y_test_bin = lb.transform(data_pack["y_test"])
                y_proba = data_pack["y_pred_proba"]

                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                for i, class_name in enumerate(encoder.classes_):
                    if y_test_bin[:, i].sum() > 0:
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        ax_roc.plot(
                            fpr, tpr, label=f"{class_name} (AUC={auc(fpr, tpr):.2f})"
                        )
                ax_roc.plot([0, 1], [0, 1], "k--")
                ax_roc.legend()
                ax_roc.set_title("ROC Curves")
                figures.append(fig_roc)

                fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
                for i, class_name in enumerate(encoder.classes_):
                    if y_test_bin[:, i].sum() > 0:
                        p, r, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                        ax_pr.plot(r, p, label=class_name)
                ax_pr.legend()
                ax_pr.set_title("Precision-Recall")
                figures.append(fig_pr)
            else:
                for _ in range(2):
                    fig_empty, ax_empty = plt.subplots(figsize=(6, 4))
                    ax_empty.text(
                        0.5,
                        0.5,
                        "Not enough data",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="gray",
                    )
                    ax_empty.axis("off")
                    figures.append(fig_empty)

            # Data distribution
            # Data distribution as Pie Chart
            fig_dist, ax_dist = plt.subplots(figsize=(6, 6))
            dist = data_pack["distribution"]
            ax_dist.pie(
                dist.values,
                labels=dist.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("pastel")
            )
            ax_dist.set_title("Stress Level Distribution")
            ax_dist.axis("equal")  # ensures circle shape
            figures.append(fig_dist)


        except Exception as e:
            print(f"ERROR in create_visuals: {e}")
            traceback.print_exc()
        return figures

    def run_pipeline(self):
        try:
            def update_callback(event_type, data):
                if event_type == "headers":
                    self._setup_treeview_headers(data)
                elif event_type == "row":
                    self.master.after(0, self._update_gui, data)

            # 1. Load + preprocess
            t0 = log_step("Loading dataset")
            df, df_numeric = load_and_preprocess_slowly(self.file_path, update_callback)
            self.df = df
            log_step("Dataset loaded", t0)

            # 2. Train
            t2 = log_step("Training models")
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(df["Stress_Level"])
            data_pack = train_and_get_data(df_numeric, y_encoded, encoder, df["Stress_Level"])
            log_step("Training done", t2)

            # 3. Visuals
            figures = self.create_visuals(data_pack, encoder)

            # 4. Show results
            self.master.after(0, lambda: self.on_processing_complete(data_pack, figures))

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))
            self.master.after(0, self.master.destroy)

    def on_processing_complete(self, data_pack, figures):
        self.master.withdraw()
        ResultsWindow(self.master, data_pack, figures, full_df=self.df)


# ----------------------- Main -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <file.csv>")
        sys.exit(1)
    root = tk.Tk()
    app = ProcessingWindow(root, sys.argv[1])
    root.mainloop()
