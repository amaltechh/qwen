import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from datetime import datetime
import os


class ResultsWindow:
    """
    A professional results dashboard window to display the stress assessment results.
    """

    def __init__(self, master, total_score, stress_level, answers_dict, questions):
        self.master = master
        self.total_score = total_score
        self.stress_level = stress_level
        self.answers_dict = answers_dict
        self.questions = questions

        # --- Window Setup ---
        self.top = tk.Toplevel(master)
        self.top.title("📊 Your Stress Assessment Report")
        self.top.geometry("1100x750")

        style = ttk.Style(self.top)
        style.theme_use("clam")
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure(
            "Level.TLabel",
            font=("Segoe UI", 24, "bold"),
            foreground=self.get_stress_color(),
        )
        style.configure("Advice.TLabel", font=("Segoe UI", 11))

        # --- Main Layout ---
        main_frame = ttk.Frame(self.top, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Header ---
        ttk.Label(
            main_frame, text="Your Assessment Results", style="Header.TLabel"
        ).pack(pady=(0, 20))

        # --- Top Section: Score and Gauge ---
        top_section = ttk.Frame(main_frame)
        top_section.pack(fill=tk.X, expand=True)

        score_frame = ttk.LabelFrame(top_section, text="Final Result", padding=15)
        score_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        ttk.Label(
            score_frame, text="Your calculated stress level is:", font=("Segoe UI", 12)
        ).pack(anchor="center", pady=5)
        ttk.Label(
            score_frame, text=self.stress_level.upper(), style="Level.TLabel"
        ).pack(anchor="center", pady=(5, 20))

        gauge_frame = ttk.LabelFrame(top_section, text="Your Score Gauge", padding=15)
        gauge_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self.create_score_gauge(gauge_frame)

        # --- Middle Section: Advice ---
        advice_frame = ttk.LabelFrame(
            main_frame, text="Personalized Advice", padding=15
        )
        advice_frame.pack(fill=tk.X, expand=True, pady=20)

        advice_text = self.get_advice()
        ttk.Label(
            advice_frame,
            text=advice_text,
            style="Advice.TLabel",
            wraplength=1000,
            justify=tk.LEFT,
        ).pack(anchor="w")

        # --- Bottom Section: Answer Breakdown Chart ---
        chart_frame = ttk.LabelFrame(
            main_frame, text="Your Answer Breakdown", padding=15
        )
        chart_frame.pack(fill=tk.BOTH, expand=True)
        self.create_answer_chart(chart_frame)

        # --- Save Button ---
        save_button = ttk.Button(
            main_frame,
            text="💾 Save Results to CSV",
            command=self.save_results,
            padding=10,
        )
        save_button.pack(pady=(20, 0))

    def get_stress_color(self):
        """Returns a color based on the stress level for styling."""
        return {
            "No stress": "green",
            "Low stress": "blue",
            "Medium stress": "orange",
            "High stress": "red",
        }.get(self.stress_level, "black")

    def get_advice(self):
        """Returns personalized advice based on the stress level."""
        return {
            "No stress": "Excellent! Your responses suggest you have very effective stress management strategies. Continue prioritizing your well-being, engaging in hobbies, and maintaining your healthy habits.",
            "Low stress": "Good job. You seem to be managing stress well overall. Pay attention to the specific areas that scored higher in the breakdown chart. Focusing on small improvements in those areas can be beneficial.",
            "Medium stress": "Your stress level is moderate. This is common, but it's a good time to act. Consider implementing more relaxation techniques like mindfulness or exercise. The 'Answer Breakdown' chart can help you identify your main stressors.",
            "High stress": "Your responses indicate a high level of stress, which can impact your health and academic performance. It is strongly recommended that you talk to a friend, family member, or a professional counselor. Your institution likely has free counseling services available.",
        }.get(self.stress_level, "Could not determine advice for your stress level.")

    def create_score_gauge(self, parent):
        """Creates a Matplotlib horizontal bar to act as a score gauge."""
        fig = Figure(figsize=(5, 1.5), dpi=100)
        ax = fig.add_subplot(111)

        min_score = len(self.questions) * 1
        max_score = len(self.questions) * 5

        ax.set_xlim(min_score, max_score)
        ax.set_ylim(0, 1)

        # Create color gradient bar
        ax.axvspan(
            min_score,
            min_score + (max_score - min_score) * 0.25,
            color="green",
            alpha=0.6,
        )
        ax.axvspan(
            min_score + (max_score - min_score) * 0.25,
            min_score + (max_score - min_score) * 0.5,
            color="blue",
            alpha=0.6,
        )
        ax.axvspan(
            min_score + (max_score - min_score) * 0.5,
            min_score + (max_score - min_score) * 0.75,
            color="orange",
            alpha=0.6,
        )
        ax.axvspan(
            min_score + (max_score - min_score) * 0.75,
            max_score,
            color="red",
            alpha=0.6,
        )

        # Plot user score
        ax.plot(
            [self.total_score, self.total_score],
            [0, 1],
            color="black",
            linewidth=3,
            marker="^",
            markersize=12,
            markevery=(1, 2),
        )
        ax.text(
            self.total_score,
            0.45,
            f" {self.total_score}",
            color="black",
            fontweight="bold",
            ha="center",
        )

        ax.set_yticks([])
        ax.set_xticks([min_score, max_score])
        ax.set_xticklabels(["Min Score", "Max Score"])
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_answer_chart(self, parent):
        """Creates a Matplotlib bar chart of the user's answers."""
        fig = Figure(figsize=(10, 5), dpi=100)
        ax = fig.add_subplot(111)

        q_labels = [f"Q{i+1}" for i in range(len(self.questions))]
        answers = list(self.answers_dict.values())

        ax.barh(q_labels, answers, color="skyblue")
        ax.set_xlabel("Score (1=Disagree, 5=Agree)")
        ax.set_title("Your Answers per Question")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.invert_yaxis()  # Display Q1 at the top
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results(self):
        """Saves the current result to a CSV file."""
        filename = "my_stress_results.csv"
        new_data = {
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Total_Score": [self.total_score],
            "Stress_Level": [self.stress_level],
        }
        df_new = pd.DataFrame(new_data)

        try:
            if os.path.exists(filename):
                # Append without header
                df_new.to_csv(filename, mode="a", header=False, index=False)
            else:
                # Create new file with header
                df_new.to_csv(filename, mode="w", header=True, index=False)

            messagebox.showinfo(
                "Success", f"Your results have been saved to {filename}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not save the file.\nError: {e}")


class StressTestApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Stress Level Assessment Tool 🧠")
        self.master.geometry("850x700")

        style = ttk.Style(self.master)
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Question.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("TRadiobutton", font=("Segoe UI", 10))

        self.questions = [
            "I often feel overwhelmed by academic workload.",
            "I feel anxious before exams or tests.",
            "I struggle to meet assignment deadlines.",
            "I worry about maintaining a high GPA or academic performance.",
            "I find it difficult to balance projects, labs, and regular classes.",
            "I feel emotionally exhausted by the demands of my course.",
            "I often feel demotivated or hopeless about my future.",
            "I experience mood swings or irritability.",
            "I feel isolated or lonely despite being surrounded by peers.",
            "I have limited time to spend with family or friends.",
            "I feel pressure to meet expectations from my family.",
            "I find it difficult to maintain healthy relationships due to academics.",
            "I hesitate to seek help when I feel stressed.",
            "I feel like others expect too much from me.",
            "I often suffer from headaches, fatigue, or body aches.",
            "I notice changes in my appetite or eating habits during stress.",
            "I feel tired even after getting enough sleep.",
            "I experience rapid heartbeat or sweating when stressed.",
            "I am unable to concentrate during lectures or study sessions.",
            "I don't have effective strategies to manage my stress.",
            "I do not engage in physical activities or hobbies to reduce stress.",
            "I do not talk to friends, faculty, or counselors when I feel overwhelmed.",
            "I don't find time for relaxation or meditation despite my busy schedule.",
            "I am not aware of counseling services offered by the institution.",
        ]
        self.options = [
            "Strongly Disagree",
            "Disagree",
            "Neutral",
            "Agree",
            "Strongly Agree",
        ]
        self.likert_mapping = {name: i + 1 for i, name in enumerate(self.options)}
        self.answers_vars = []

        self.create_widgets()

    def create_scrollable_frame(self):
        main_frame = ttk.Frame(self.master, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(main_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_widgets(self):
        ttk.Label(
            self.master,
            text="Stress Assessment Questionnaire",
            style="Header.TLabel",
            padding=(10, 10, 10, 0),
        ).pack()
        self.create_scrollable_frame()
        for i, question_text in enumerate(self.questions):
            q_frame = ttk.LabelFrame(
                self.scrollable_frame, text=f"Question {i+1}", padding=15
            )
            q_frame.pack(fill=tk.X, padx=10, pady=7)
            ttk.Label(
                q_frame,
                text=question_text,
                wraplength=750,
                justify=tk.LEFT,
                style="Question.TLabel",
            ).pack(anchor="w")
            answer_var = tk.IntVar(value=0)
            self.answers_vars.append(answer_var)
            options_frame = ttk.Frame(q_frame)
            options_frame.pack(pady=10, anchor="w")
            for name, value in self.likert_mapping.items():
                ttk.Radiobutton(
                    options_frame, text=name, variable=answer_var, value=value
                ).pack(side=tk.LEFT, padx=15, anchor="w")
        submit_button = ttk.Button(
            self.master,
            text="✅ Calculate My Stress Level",
            command=self.calculate_results,
            padding=10,
        )
        submit_button.pack(pady=20)

    def classify_stress(self, total_score):
        min_score = len(self.questions) * 1
        max_score = len(self.questions) * 5
        score_range = max_score - min_score
        if score_range == 0:
            return "Medium stress"
        normalized_score = (total_score - min_score) / score_range
        if normalized_score <= 0.25:
            return "No stress"
        elif normalized_score <= 0.50:
            return "Low stress"
        elif normalized_score <= 0.75:
            return "Medium stress"
        else:
            return "High stress"

    def calculate_results(self):
        total_score = 0
        answers_dict = {}
        for i, answer_var in enumerate(self.answers_vars):
            if answer_var.get() == 0:
                messagebox.showwarning(
                    "Incomplete Form",
                    f"Please answer Question {i+1} before submitting.",
                )
                return
            score = answer_var.get()
            total_score += score
            answers_dict[f"Q{i+1}"] = score

        stress_level = self.classify_stress(total_score)

        # Launch the new results window instead of a simple messagebox
        self.master.withdraw()  # Hide the main quiz window
        ResultsWindow(
            self.master, total_score, stress_level, answers_dict, self.questions
        )


def main():
    root = tk.Tk()
    app = StressTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
