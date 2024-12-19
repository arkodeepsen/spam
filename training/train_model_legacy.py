import numpy as np
import pandas as pd
import nltk, string, logging, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_text(text):
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

def plot_dataset_insights(df):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    sns.histplot(data=df, x='num_characters', hue='target', bins=50)
    plt.title('Message Length Distribution')
    
    plt.subplot(132)
    df['target'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    
    plt.subplot(133)
    sns.boxplot(data=df, x='target', y='num_words')
    plt.title('Word Count by Class')
    
    plt.tight_layout()
    plt.savefig('./graphs/dataset_insights.png')
    plt.close()

def plot_word_clouds(df):
    from wordcloud import WordCloud
    plt.figure(figsize=(15, 5))
    
    # Map text labels to numeric
    df['target_num'] = df['target'].map({'ham': 0, 'spam': 1})
    
    for idx, label in enumerate(['ham', 'spam']):
        # Get text for current label
        text = ' '.join(df[df['target'] == label]['transformed_text'])
        
        if not text.strip():
            logger.warning(f"No text found for label: {label}")
            continue
            
        try:
            wordcloud = WordCloud(width=800, height=400).generate(text)
            plt.subplot(1, 2, idx+1)
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.title(f'Word Cloud - {label.upper()}')
        except Exception as e:
            logger.error(f"Error generating wordcloud for {label}: {e}")
    
    plt.savefig('./graphs/wordclouds.png')
    plt.close()

def plot_performance_metrics(y_test, y_pred, model):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    plt.subplot(132)
    performance_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision'],
        'Score': [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred)]
    })
    sns.barplot(x='Metric', y='Score', data=performance_df)
    plt.title('Model Performance')
    
    plt.subplot(133)
    etc = model.named_estimators_['et']
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
        logger.info("Loading data...")
        df = pd.read_csv('./data/spam.csv', encoding='latin-1')
        df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        df = df.rename(columns={'v1': 'target', 'v2': 'text'})
        
        logger.info(f"Target value counts:\n{df['target'].value_counts()}")
        
        # Add numerical features
        df['num_characters'] = df['text'].apply(len)
        df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
        df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
        
        logger.info("Transforming text...")
        df['transformed_text'] = df['text'].apply(transform_text)
        
        # Verify transformed text
        logger.info(f"Sample transformed text:\n{df['transformed_text'].head()}")
        
        logger.info("Generating visualizations...")
        plot_dataset_insights(df)
        plot_word_clouds(df)
        
        # Text vectorization
        tfidf = TfidfVectorizer(max_features=3000)
        X = tfidf.fit_transform(df['transformed_text']).toarray()
        # Convert target to numeric for model
        y = (df['target'] == 'spam').astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
        
        # Create ensemble
        logger.info("Training model...")
        svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
        mnb = MultinomialNB()
        etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
        
        voting = VotingClassifier([('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')
        voting.fit(X_train, y_train)
        
        y_pred = voting.predict(X_test)
        
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred)
        }
        
        save_metrics(metrics)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        plot_performance_metrics(y_test, y_pred, voting)
        
        logger.info("Saving models...")
        pickle.dump(tfidf, open('./models/vectorizer.pkl', 'wb'))
        pickle.dump(voting, open('./models/model.pkl', 'wb'))
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        main()
    except Exception as e:
        print(f"Fatal error: {e}")