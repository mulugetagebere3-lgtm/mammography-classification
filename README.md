import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. ዳታ ንምንባብ
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. ጽሑፍ ናብ ቁጽሪ ንምቕያር (Vectorization)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['report'])
y = df['target']

# 3. ንምምሃርን ንምፍታንን ምክፋል (Split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ሞዴል ምምሃር (Training)
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Accuracy ምርኣይ
preds = model.predict(X_val)
print(f"ሞዴል Accuracy: {accuracy_score(y_val, preds) * 100:.2f}%")

# 6. ኣብቲ Test Data (መልሲ ዘይብሉ) Prediction ምስራሕ
X_test = vectorizer.transform(test_df['report'])
test_preds = model.predict(X_test)

# 7. ናብ Submission ፋይል ምቕያር
submission = pd.DataFrame({'ID': test_df['ID'], 'target': test_preds})
submission.to_csv('final_submission.csv', index=False)
print("--- 'final_submission.csv' ብዓወት ተፈጢሩ ኣሎ! ---")
