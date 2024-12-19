import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import os, sys, io, traceback, threading, torch
from contextlib import redirect_stdout
from training.train_model_lite import main as train_lite
from training.train_model_legacy import main as train_legacy  
from training.train_model_mbo import main as train_mbo
from torch.cuda import is_available as cuda_available

class SpamDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Detection Model Trainer")
        self.root.geometry("1200x800")
        
        # Create tabs
        self.tabControl = ttk.Notebook(root)
        self.train_tab = ttk.Frame(self.tabControl)
        self.viz_tab = ttk.Frame(self.tabControl)
        self.log_tab = ttk.Frame(self.tabControl)
        self.current_viz = 0
        self.viz_paths = []
        
        self.tabControl.add(self.train_tab, text='Train Model')
        self.tabControl.add(self.viz_tab, text='Visualizations')
        self.tabControl.add(self.log_tab, text='Logs')
        self.tabControl.pack(expand=1, fill="both")
        
        self.setup_training_tab()
        self.setup_visualization_tab()
        self.setup_log_tab()
        
        # Create folders if they don't exist
        for folder in ['./graphs', './models']:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def setup_training_tab(self):
        # Training controls
        control_frame = ttk.LabelFrame(self.train_tab, text="Training Controls", padding=10)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Add model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        self.model_var = tk.StringVar(value="lite")
        ttk.Radiobutton(model_frame, text="Lite Model", 
                       value="lite", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Legacy Model", 
                       value="legacy", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Monarch Butterfly", 
                       value="mbo", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        
                # Add MBO parameters frame
        self.mbo_frame = ttk.LabelFrame(control_frame, text="MBO Parameters", padding=10)
        
        # MBO parameter entries
        params = {
            'n_butterflies': (20, 5, 100),  # default, min, max
            'p_period': (1.2, 0.1, 5.0),
            'migration_ratio': (0.85, 0.1, 1.0),
            'max_iter': (30, 10, 100)
        }
        
        self.mbo_params = {}
        
        for param, (default, min_val, max_val) in params.items():
            frame = ttk.Frame(self.mbo_frame)
            frame.pack(fill='x', padx=5, pady=2)
            
            ttk.Label(frame, text=f"{param}:").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.pack(side=tk.LEFT, padx=5)
            
            self.mbo_params[param] = {
                'var': var,
                'min': min_val,
                'max': max_val
            }
        
        # GPU checkbox
        self.use_gpu = tk.BooleanVar(value=cuda_available())
        self.gpu_check = ttk.Checkbutton(
            self.mbo_frame, 
            text="Use GPU (if available)", 
            variable=self.use_gpu,
            state='normal' if cuda_available() else 'disabled'
        )
        self.gpu_check.pack(pady=5)
        
        # Show/hide MBO parameters based on model selection
        def on_model_change(*args):
            if self.model_var.get() == "mbo":
                self.mbo_frame.pack(fill="x", padx=5, pady=5, after=model_frame)
            else:
                self.mbo_frame.pack_forget()
                
        self.model_var.trace_add('write', on_model_change)
        
        # Metrics display
        self.metrics_frame = ttk.LabelFrame(self.train_tab, text="Model Metrics", padding=10)
        self.metrics_frame.pack(fill="x", padx=5, pady=5)
        
        self.metrics_text = scrolledtext.ScrolledText(self.metrics_frame, height=5)
        self.metrics_text.pack(fill="both", expand=True)
        
        self.train_button = ttk.Button(control_frame, text="Train Model", command=self.start_training)
        self.train_button.pack(pady=5)
        
        self.progress = ttk.Progressbar(control_frame, length=400, mode='indeterminate')
        self.progress.pack(pady=5)
        
    def setup_visualization_tab(self):
        # Main frame with scrollbar
        main_frame = ttk.Frame(self.viz_tab)
        main_frame.pack(fill="both", expand=True)
        
        # Add canvas with scrollbar
        self.viz_canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.viz_canvas.yview)
        self.viz_frame = ttk.Frame(self.viz_canvas)
        
        self.viz_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.viz_canvas.pack(side="left", fill="both", expand=True)
        
        # Create window in canvas
        self.canvas_frame = self.viz_canvas.create_window((0, 0), window=self.viz_frame, anchor="nw")
        
        # Configure scrolling
        self.viz_frame.bind("<Configure>", self.on_frame_configure)
        self.viz_canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Image frames
        self.insights_frame = ttk.LabelFrame(self.viz_frame, text="Dataset Insights")
        self.wordcloud_frame = ttk.LabelFrame(self.viz_frame, text="Word Clouds")
        self.metrics_frame = ttk.LabelFrame(self.viz_frame, text="Performance Metrics")
        
        self.insights_frame.pack(fill="x", padx=5, pady=5)
        self.wordcloud_frame.pack(fill="x", padx=5, pady=5)
        self.metrics_frame.pack(fill="x", padx=5, pady=5)

    def on_frame_configure(self, event=None):
        self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox("all"))
        
    def on_canvas_configure(self, event):
        # Update the width of the canvas window
        width = event.width
        self.viz_canvas.itemconfig(self.canvas_frame, width=width)

    def prev_viz(self):
        if self.current_viz > 0:
            self.current_viz -= 1
            self.show_current_viz()
    
    def next_viz(self):
        if self.current_viz < len(self.viz_paths) - 1:
            self.current_viz += 1
            self.show_current_viz()
    
    def show_current_viz(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
            
        if self.viz_paths:
            self.display_image(self.viz_paths[self.current_viz], self.image_frame)
            
        # Update button states
        self.prev_button["state"] = "normal" if self.current_viz > 0 else "disabled"
        self.next_button["state"] = "normal" if self.current_viz < len(self.viz_paths) - 1 else "disabled"
        
    def setup_log_tab(self):
        self.log_text = scrolledtext.ScrolledText(self.log_tab)
        self.log_text.pack(fill="both", expand=True)
    
    def start_training(self):
        self.train_button.config(state='disabled')
        self.progress.start()
        self.clear_displays()
        
        # Start training in a separate thread
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    
    def train_model(self):
        log_capture = io.StringIO()
        with redirect_stdout(log_capture):
            try:
                model_type = self.model_var.get()
                if model_type == "lite":
                    train_lite()
                elif model_type == "legacy":
                    train_legacy()
                else:
                    # Validate and get MBO parameters
                    mbo_params = {}
                    for param, config in self.mbo_params.items():
                        try:
                            value = float(config['var'].get())
                            if not config['min'] <= value <= config['max']:
                                raise ValueError(
                                    f"{param} must be between {config['min']} and {config['max']}"
                                )
                            mbo_params[param] = value
                        except ValueError as e:
                            raise ValueError(f"Invalid {param}: {str(e)}")
                    
                    mbo_params['use_gpu'] = self.use_gpu.get()
                    train_mbo(mbo_params)
                    
                self.root.after(0, self.update_displays)
            except Exception:
                error_msg = traceback.format_exc()
                self.root.after(0, lambda: self.handle_error(error_msg))
            finally:
                log_text = log_capture.getvalue()
                self.root.after(0, lambda: self.update_log(log_text))
                self.root.after(0, self.finish_training)

    def handle_error(self, error_msg):
        self.log_text.insert(tk.END, f"Error occurred:\n{error_msg}\n")
        messagebox.showerror("Error", "Training failed. Check logs for details.")

    def update_log(self, log_text):
        self.log_text.insert(tk.END, f"Selected Model: {self.model_var.get().upper()}\n")
        self.log_text.insert(tk.END, log_text)
        self.log_text.see(tk.END)
    
    def update_displays(self):
        # Clear previous displays
        for frame in [self.insights_frame, self.wordcloud_frame, self.metrics_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        # Display all visualizations
        if os.path.exists('./graphs/dataset_insights.png'):
            self.display_image('./graphs/dataset_insights.png', self.insights_frame)
        if os.path.exists('./graphs/wordclouds.png'):
            self.display_image('./graphs/wordclouds.png', self.wordcloud_frame)
        if os.path.exists('./graphs/performance_metrics.png'):
            self.display_image('./graphs/performance_metrics.png', self.metrics_frame)
        
        # Update metrics
        try:
            with open('./models/metrics.txt', 'r') as f:
                metrics = f.read()
                self.metrics_text.delete(1.0, tk.END)
                self.metrics_text.insert(tk.END, metrics)
        except Exception as e:
            self.log_text.insert(tk.END, f"Error reading metrics: {e}\n")

    def clear_displays(self):
        self.metrics_text.delete(1.0, tk.END)
        self.log_text.delete(1.0, tk.END)
        for frame in [self.insights_frame, self.wordcloud_frame, self.metrics_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

    def display_image(self, image_path, frame):
        try:
            img = Image.open(image_path)
            # Adjust size while maintaining aspect ratio
            display_width = 1000
            ratio = display_width / float(img.size[0])
            display_height = int(float(img.size[1]) * float(ratio))
            
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            label = ttk.Label(frame, image=photo)
            label.image = photo
            label.pack(padx=5, pady=5)
        except Exception as e:
            self.log_text.insert(tk.END, f"Error displaying image {image_path}: {e}\n")
            
    def finish_training(self):
        self.progress.stop()
        self.train_button.config(state='normal')

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SpamDetectionUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)