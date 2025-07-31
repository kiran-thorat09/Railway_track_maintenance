#Installing require libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

#Load data
df = pd.read_csv('train_data_clean.xls')
#Drop irrelevant columns based on correlation matrix(EDA)
irrelevant_cols = ['segment_id', 'install_year', 'train_load_tons', 'failure_cause','maintenance_required', 'weather','rainfall_mm']
df = df.drop(columns=irrelevant_cols)
#Checking columns
# for col in df.columns:
#     print(col)

#Date time conversion
df['last_maintenance'] = pd.to_datetime(df['last_maintenance'], errors='coerce')

#Calculate months on the basis of last maintenance
df['last_maintenance_months'] = ((pd.Timestamp.now() - df['last_maintenance']) / pd.Timedelta(days=30)).astype(int)

#Drop the original last_maintenance column
df.drop('last_maintenance', axis=1, inplace=True)

#Round numeric values to 2 decimal
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'failure':
        df[col] = df[col].round(2)

#Split features and target
X = df.drop(columns='failure')
y = df['failure']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#print("--------------")

#Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Save model for future use
joblib.dump(model, "railway_track_model.pkl")
