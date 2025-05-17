import pandas as pd
import os
from sklearn.model_selection import train_test_split

wd = os.getcwd()
print(wd)
data_path = "../data/"
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

def get_train_text_query_pair(method : str = 'random', num_pair: int = 5):
    if method == 'random':
        return df_train.sample(num_pair)
    elif method == 'cluster':
        cluster_df = pd.read_csv('cluster_data/best_representatives.csv')
        return cluster_df.head(num_pair)
    return

def get_test_text_query_pair(num_pair: int = 5):
    return df_test.sample(1)

# function has to take two params
# n = 1, 2... 50
# partition: has a couple of examples ABAB
# returns n from partition
# test_set: a set with tests queries 

# TODO: for loop that wraps this thing.


def get_model_input(method : str = 'random', num_train: int = 5, num_test: int = 1):
    if method not in ('random', 'cluster'):
        raise ValueError("Method must be either 'random' or 'cluster'")
    df_prompt = get_train_text_query_pair(method=method, num_pair=num_train)
    df_test_in = get_test_text_query_pair(num_pair=num_test)
    input_str = ""
    for i in range(len(df_prompt)):
        input_str += f"[EXAMPLE NL QUESTION {i+1}] "
        input_str += df_prompt.iloc[i]["natural_language_query"]
        input_str += "\n"
        input_str += f"[EXAMPLE SQL PROMPT {i+1}] "
        input_str += df_prompt.iloc[i]["sql_query"]
        input_str += "\n\n"

    input_str += f"[NL QUESTION] "
    input_str += df_test_in.iloc[0]["natural_language_query"]
    input_str += f"\n[SQL PROMPT] "
    input_str = """
    You are a strict SQL translator. \n
    You MUST ONLY output a single SQL query. \n
    You are NOT allowed to repeat the instructions, the examples, or any natural language text. \n
    You CANNOT explain anything. \n
    You MUST copy the style of the EXAMPLES exactly.

    Respond ONLY with the SQL query.
    \n---\n
    """ + input_str
    return [input_str, df_test_in.iloc[0]["sql_query"]]