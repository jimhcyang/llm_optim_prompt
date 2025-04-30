import pandas as pd
import os
from sklearn.model_selection import train_test_split

data_path = "../../data/"
file_name = "atis_text_to_sql_pairs.csv"

file_path = os.path.join(data_path, file_name)

atis_df = pd.read_csv(file_path)

# Step 1: Read the CSV file
df = atis_df.copy()

# Step 2: Define features and target
X = df.drop('sql_query', axis=1)  # replace 'target_column' with the name of your target variable
y = df['sql_query']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Optional: print shapes to verify
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

output_dir = 'split_data'
os.makedirs(output_dir, exist_ok=True)

# Save the split datasets to CSV files
train_path = os.path.join(output_dir, 'train.csv')
test_path = os.path.join(output_dir, 'test.csv')

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)