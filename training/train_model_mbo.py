import numpy as np
import pandas as pd
import nltk, string, logging, pickle, torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch.cuda import is_available as cuda_available

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonarchButterflyOptimizer:
    def __init__(self, bounds, n_butterflies=20, p_period=1.2, migration_ratio=0.85, max_iter=30, use_gpu=False):
        self.bounds = bounds
        self.n_butterflies = n_butterflies
        self.p_period = p_period
        self.migration_ratio = migration_ratio
        self.max_iter = max_iter
        self.best_solution = None
        self.best_fitness = float('-inf')
        
        # GPU setup
        self.use_gpu = use_gpu and cuda_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")

    def initialize(self):
        try:
            population = []
            for _ in range(self.n_butterflies):
                butterfly = {}
                for param, (low, high) in self.bounds.items():
                    if isinstance(low, int) and isinstance(high, int):
                        butterfly[param] = int(torch.randint(low, high+1, (1,), device=self.device).item())
                    else:
                        butterfly[param] = float(torch.rand(1, device=self.device).item() * (high - low) + low)
                population.append(butterfly)
            return population
        except RuntimeError as e:
            logger.error(f"CUDA error during initialization: {e}")
            self.device = torch.device('cpu')
            logger.info("Falling back to CPU")
            return self.initialize()

    def migration(self, population):
        try:
            new_population = []
            migration_tensor = torch.rand(len(population), device=self.device)
            
            for idx, butterfly in enumerate(population):
                if migration_tensor[idx].item() < self.migration_ratio:
                    new_butterfly = {}
                    for param in butterfly:
                        r = torch.rand(1, device=self.device).item()
                        new_val = butterfly[param] + self.p_period * r * (self.best_solution[param] - butterfly[param])
                        new_butterfly[param] = self.clip(new_val, param)
                    new_population.append(new_butterfly)
                else:
                    new_population.append(butterfly.copy())
            return new_population
        except RuntimeError as e:
            logger.error(f"CUDA error during migration: {e}")
            self.device = torch.device('cpu')
            logger.info("Falling back to CPU")
            return self.migration(population)
    
    def clip(self, value, param):
        low, high = self.bounds[param]
        if isinstance(low, int) and isinstance(high, int):
            return int(np.clip(value, low, high))
        return np.clip(value, low, high)
    
    def optimize(self, fitness_func):
        population = self.initialize()
        
        for _ in range(self.max_iter):
            for butterfly in population:
                fitness = fitness_func(butterfly)
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = butterfly.copy()
            
            population = self.migration(population)
        
        return self.best_solution, self.best_fitness
    
def plot_dataset_insights(df):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.histplot(data=df, x='feature_length', hue='target', bins=50)
    plt.title('Message Length Distribution')
    
    plt.subplot(132)
    df['target'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    
    plt.subplot(133)
    sns.boxplot(data=df, x='target', y='word_count')
    plt.title('Word Count by Class')
    
    plt.tight_layout()
    plt.savefig('./graphs/dataset_insights.png')
    plt.close()

def plot_word_clouds(df):
    from wordcloud import WordCloud
    plt.figure(figsize=(15, 5))
    
    for idx, label in enumerate(['ham', 'spam']):
        text = ' '.join(df[df['target'] == label]['transformed_text'])
        wordcloud = WordCloud(width=800, height=400).generate(text)
        
        plt.subplot(1, 2, idx+1)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title(f'Word Cloud - {label.upper()}')
    
    plt.savefig('./graphs/wordclouds.png')
    plt.close()

def plot_performance_metrics(y_test, y_pred, model):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    plt.subplot(132)
    report = classification_report(y_test, y_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='RdYlGn')
    plt.title('Classification Report')
    
    plt.subplot(133)
    etc = model.named_estimators_['etc']
    importances = pd.Series(etc.feature_importances_)
    importances.nlargest(10).plot(kind='bar')
    plt.title('Top 10 Important Features')
    
    plt.tight_layout()
    plt.savefig('./graphs/performance_metrics.png')
    plt.close()

def save_metrics(metrics):
    with open('./models/metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

def create_optimized_ensemble(X_train, y_train, mbo_params):
    param_bounds = {
        'svc_C': (0.1, 20.0),
        'svc_gamma': (0.001, 1.0),
        'mnb_alpha': (0.1, 2.0),
        'etc_n_estimators': (100, 300),
        'w1': (0, 5),
        'w2': (0, 5),
        'w3': (0, 5)
    }
    
    mbo = MonarchButterflyOptimizer(
        param_bounds,
        n_butterflies=int(mbo_params.get('n_butterflies', 20)),
        p_period=float(mbo_params.get('p_period', 1.2)),
        migration_ratio=float(mbo_params.get('migration_ratio', 0.85)),
        max_iter=int(mbo_params.get('max_iter', 30)),
        use_gpu=bool(mbo_params.get('use_gpu', False))
    )
    
    def fitness_function(params):
        svc = SVC(kernel='rbf', C=params['svc_C'], 
                  gamma=params['svc_gamma'], probability=True)
        mnb = MultinomialNB(alpha=params['mnb_alpha'])
        etc = ExtraTreesClassifier(n_estimators=int(params['etc_n_estimators']))
        
        estimators = [('svc', svc), ('mnb', mnb), ('etc', etc)]
        weights = [params['w1'], params['w2'], params['w3']]
        
        clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        return np.mean(scores)
    
    # Initialize and run MBO
    mbo = MonarchButterflyOptimizer(param_bounds)
    best_params, _ = mbo.optimize(fitness_function)
    # Create final model with optimized parameters
    svc = SVC(kernel='rbf', C=best_params['svc_C'], 
              gamma=best_params['svc_gamma'], probability=True)
    mnb = MultinomialNB(alpha=best_params['mnb_alpha'])
    etc = ExtraTreesClassifier(n_estimators=int(best_params['etc_n_estimators']))
    
    estimators = [('svc', svc), ('mnb', mnb), ('etc', etc)]
    weights = [best_params['w1'], best_params['w2'], best_params['w3']]
    
    return VotingClassifier(estimators=estimators, voting='soft', weights=weights)

def main(mbo_params=None):
    try:
        logger.info("Loading data...")
        # Load and preprocess data
        df = pd.read_csv('./data/spam.csv', encoding='latin-1')
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        df = df.rename(columns={'v1': 'target', 'v2': 'text'})
        
        logger.info("Preprocessing text...")
        df['transformed_text'] = df['text'].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
        df['word_count'] = df['transformed_text'].str.split().str.len()
        df['feature_length'] = df['transformed_text'].apply(len)
        
        logger.info("Generating visualizations...")
        plot_dataset_insights(df)
        plot_word_clouds(df)
        
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
        X = tfidf.fit_transform(df['transformed_text'])
        y = (df['target'] == 'spam').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info("Training model with MBO...")
        if mbo_params and mbo_params.get('use_gpu'):
            logger.info("GPU acceleration enabled")
        model = create_optimized_ensemble(X_train, y_train, mbo_params or {})
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }
        
        save_metrics(metrics)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        plot_performance_metrics(y_test, y_pred, model)
        
        logger.info("Saving models...")
        with open('./models/vectorizer_mbo.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        with open('./models/model_mbo.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        logger.info("MBO optimization completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()