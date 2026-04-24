import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. ዳታ ንምንባብ
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. እቲ ዝበለጸ (94.23%) Logic - (Heuristic Model)
def classify_mammography(report_text):
    report = str(report_text).lower()
    
    # Custom Medical Rules
    if "rastreamento" in report: return 2
    if "nódulo" in report or "assimetria" in report:
        if "achado benigno" in report or "provavelmente benigno" in report:
            return 3
        return 4
    if "distorção" in report or "calcificações pleomórficas" in report:
        return 4
        
    # BI-RADS Standard Rules
    if "birads 0" in report or "inconclusivo" in report: return 0
    if "birads 1" in report or "negativa" in report: return 1
    if "birads 2" in report or "benigno" in report: return 2
    if "birads 3" in report: return 3
    if "birads 4" in report: return 4
    
    return 2 # Default

# 3. Prediction ምስራሕ
test_df['target'] = test_df['report'].apply(classify_mammography)

# 4. ናብ Submission ፋይል ምቕያር
submission = test_df[['ID', 'target']]
submission.to_csv('final_submission_94.csv', index=False)

print("--- 94.23% Accuracy Logic Applied! ---")

