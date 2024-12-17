import numpy as np
import pandas as pd
import nltk, string, logging, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")

def improved_transform_text(text):
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        
        text = str(text).lower()
        words = nltk.word_tokenize(text)
        
        words = [lemmatizer.lemmatize(word) for word in words 
                if word.isalnum() and 
                word not in stopwords.words('english') and 
                word not in string.punctuation]
        
        return " ".join(words)
    except Exception as e:
        logger.error(f"Error in text transformation: {e}")
        return text

def extract_features(df):
    try:
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).split())))
        df['uppercase_count'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
        df['special_char_count'] = df['text'].apply(lambda x: sum(not c.isalnum() for c in str(x)))
        return df
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        return df

def create_optimized_ensemble():
    try:
        svc = SVC(kernel='rbf', C=10, gamma='auto', probability=True, random_state=42)
        mnb = MultinomialNB(alpha=0.1)
        etc = ExtraTreesClassifier(n_estimators=200, max_depth=None, 
                                 min_samples_split=2, random_state=42)
        
        estimators = [('svc', svc), ('mnb', mnb), ('etc', etc)]
        voting_clf = VotingClassifier(estimators=estimators, 
                                    voting='soft', 
                                    weights=[2,1,2])
        return voting_clf
    except Exception as e:
        logger.error(f"Error creating ensemble: {e}")
        raise

def plot_dataset_insights(df):
    plt.figure(figsize=(15, 5))
    
    # Message length distribution
    plt.subplot(131)
    sns.histplot(data=df, x='text_length', hue='target', bins=50)
    plt.title('Message Length Distribution')
    
    # Class distribution
    plt.subplot(132)
    df['target'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    
    # Word count distribution
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
    # Confusion Matrix
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    # Classification Report Visualization
    plt.subplot(132)
    report = classification_report(y_test, y_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='RdYlGn')
    plt.title('Classification Report')
    
    # Feature Importance (for ExtraTreesClassifier)
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
    
def main():
    try:
        # Load and preprocess data
        df = pd.read_csv('./data/spam.csv', encoding='latin-1')
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
        df = df.rename(columns={'v1': 'target', 'v2': 'text'})
        
        logger.info("Preprocessing text...")
        df['transformed_text'] = df['text'].apply(improved_transform_text)
        df = extract_features(df)
        
        logger.info("Generating dataset insights...")
        plot_dataset_insights(df)
        plot_word_clouds(df)
        
        # Vectorization with optimized parameters
        tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,3),
            min_df=2,
            max_df=0.95
        )
        
        X = tfidf.fit_transform(df['transformed_text'])
        y = (df['target'] == 'spam').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info("Training model...")
        model = create_optimized_ensemble()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        }
        
        # Save metrics to file
        save_metrics(metrics)
        
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        plot_performance_metrics(y_test, y_pred, model)
        
        with open('./models/vectorizer_optimized.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        with open('./models/model_optimized.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        logger.info(f"Training completed. Metrics:\n{metrics}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()