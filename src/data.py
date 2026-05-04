import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(csv_path=r"C:\Projects\Fake News Classifier\src\WELFake_Dataset.csv"):
    """
    Loads and cleans the dataset.
    """
    # Load it: Read the raw CSV file into a pandas DataFrame.
    df = pd.read_csv(csv_path)
    
    # Clean it: Search for and completely drop any rows where either the title or the text is missing
    df = df.dropna(subset=['title', 'text'])
    
    # Merge it: Create a brand new column called input_text.
    df['input_text'] = df['title'] + " " + df['text']
    
    # Trim it: Drop the subject and date columns entirely.
    df = df.drop(columns=['subject', 'date'], errors='ignore')
    
    # Validate it: Ensure the label column strictly contains 0 (fake) and 1 (real).
    df = df[df['label'].isin([0, 1, '0', '1'])]
    df['label'] = df['label'].astype(int)
    
    # Return it: Spit out a clean DataFrame containing only input_text and label.
    return df[['input_text', 'label']].copy()

def get_splits(df):
    """
    Divides the clean data into three distinct buckets: 80% Train, 10% Validation, and 10% Test.
    Uses stratified splitting to maintain the class ratio.
    """
    # The Split: 80% Train, 20% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df['label'],
        random_state=42
    )
    
    # 50% of the 20% Temp data = 10% Validation, 10% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df['label'],
        random_state=42
    )
    
    return train_df, val_df, test_df
