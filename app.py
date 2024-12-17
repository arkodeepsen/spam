import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

class SpamClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Email/SMS Spam Classifier")
        self.root.geometry("800x600")
        
        # Load models
        try:
            self.legacy_tfidf = pickle.load(open('./models/vectorizer.pkl', 'rb'))
            self.legacy_model = pickle.load(open('./models/model.pkl', 'rb'))
            self.optimized_tfidf = pickle.load(open('./models/vectorizer_optimized.pkl', 'rb'))
            self.optimized_model = pickle.load(open('./models/model_optimized.pkl', 'rb'))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            root.destroy()
            return
            
        self.setup_ui()
        
    def setup_ui(self):
        # Model selection
        model_frame = ttk.LabelFrame(self.root, text="Model Selection", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        self.model_var = tk.StringVar(value="optimized")
        ttk.Radiobutton(model_frame, text="Optimized Model", 
                       value="optimized", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="Legacy Model", 
                       value="legacy", variable=self.model_var).pack(side=tk.LEFT, padx=5)
        
        # Input area
        input_frame = ttk.LabelFrame(self.root, text="Enter Message", padding=10)
        input_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=10)
        self.input_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Predict button
        self.predict_btn = ttk.Button(self.root, text="Predict", command=self.predict)
        self.predict_btn.pack(pady=10)
        
        # Result display
        result_frame = ttk.LabelFrame(self.root, text="Prediction Result", padding=10)
        result_frame.pack(fill="x", padx=10, pady=5)
        
        self.result_label = ttk.Label(result_frame, text="", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=10)
        
    def transform_text_legacy(self, text):
        ps = PorterStemmer()
        text = text.lower()
        text = nltk.word_tokenize(text)
        
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
                
        text = y[:]
        y.clear()
        
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
                
        text = y[:]
        y.clear()
        
        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)
        
    def transform_text_optimized(self, text):
        lemmatizer = WordNetLemmatizer()
        text = str(text).lower()
        words = nltk.word_tokenize(text)
        
        words = [lemmatizer.lemmatize(word) for word in words 
                if word.isalnum() and 
                word not in stopwords.words('english') and 
                word not in string.punctuation]
        
        return " ".join(words)
    
    def predict(self):
        text = self.input_text.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter a message")
            return
            
        try:
            if self.model_var.get() == "optimized":
                transformed_text = self.transform_text_optimized(text)
                vector = self.optimized_tfidf.transform([transformed_text])
                prediction = self.optimized_model.predict(vector)[0]
            else:
                transformed_text = self.transform_text_legacy(text)
                vector = self.legacy_tfidf.transform([transformed_text])
                prediction = self.legacy_model.predict(vector)[0]
                
            result = "SPAM" if prediction == 1 else "NOT SPAM"
            color = "#ff4444" if prediction == 1 else "#44ff44"
            self.result_label.config(text=result, foreground=color)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
if __name__ == "__main__":
    try:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        
        root = tk.Tk()
        app = SpamClassifierUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")