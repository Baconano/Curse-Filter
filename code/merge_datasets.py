import pandas as pd
import os

def create_master_dataset():
    print("🚀 Starting dataset merge process...")

    # Define paths relative to the 'code' folder
    jigsaw_path = os.path.join('..', 'datasets', 'train.csv')
    davidson_path = os.path.join('..', 'datasets', 'labeled_data.csv')

    # 1. Load the Jigsaw Dataset
    if os.path.exists(jigsaw_path):
        jigsaw_df = pd.read_csv(jigsaw_path)
        jigsaw_clean = jigsaw_df[['comment_text', 'toxic']].copy()
        jigsaw_clean.columns = ['text', 'is_toxic']
        print(f"✅ Loaded Jigsaw dataset: {len(jigsaw_clean)} rows")
    else:
        print(f"❌ Error: {jigsaw_path} not found!")
        return

    # 2. Load the Davidson Dataset
    if os.path.exists(davidson_path):
        davidson_df = pd.read_csv(davidson_path)
        davidson_clean = davidson_df[['tweet', 'class']].copy()
        davidson_clean['is_toxic'] = davidson_clean['class'].map({0: 1, 1: 1, 2: 0})
        davidson_clean = davidson_clean[['tweet', 'is_toxic']]
        davidson_clean.columns = ['text', 'is_toxic']
        print(f"✅ Loaded Davidson dataset: {len(davidson_clean)} rows")
    else:
        print(f"❌ Error: {davidson_path} not found!")
        return

    # 3. Create Custom Data Injection (The "Fix" Layer)
    # This addresses the "Pickle" false positives and the "Bitch/Pussy" missing insults
    custom_data = [
        # Toxic fixes (1) - Ensuring these are caught
        ("You are a total bitch.", 1),
        ("Stop acting like a pussy.", 1),
        ("Shut the fuck up you bitch.", 1),
        ("You're a piece of shit.", 1),
        ("Get out of this server you bitch.", 1),
        ("You're actually dogshit at this game.", 1),
        ("Shut up pussy.", 1),
        
        # Context fixes (0) - Teaching the bot about non-human targets
        ("I hate pickles on my burger.", 0),
        ("That play was totally gross!", 0),
        ("I hate it when the game lags like this.", 0),
        ("This pizza tastes gross.", 0),
        ("I hate doing my homework on Sundays.", 0),
        ("That's a gross amount of damage!", 0),
        ("The weather today is gross.", 0),
        ("I hate pickles.", 0),
        ("I really hate that flavor of ice cream.", 0),
        ("The boss fight is gross but cool.", 0)
    ]
    
    # We multiply this data by 100 so it doesn't get lost in the 180k rows
    custom_df = pd.DataFrame(custom_data * 100, columns=['text', 'is_toxic'])
    print(f"✅ Created custom injection: {len(custom_df)} weighted rows")

    # 4. Merge all data
    master_df = pd.concat([jigsaw_clean, davidson_clean, custom_df], ignore_index=True)

    # Clean up: remove empty rows or duplicates
    master_df = master_df.dropna(subset=['text'])
    master_df = master_df.drop_duplicates(subset=['text'])

    # 5. Export to CSV
    output_file = 'master_training_dataset.csv'
    master_df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"🎉 Success! Master dataset saved to: {output_file}")
    print(f"Total Rows: {len(master_df)}")
    print(f"Toxic Samples: {master_df['is_toxic'].sum()}")
    print(f"Clean Samples: {len(master_df) - master_df['is_toxic'].sum()}")
    print("-" * 30)
    output_file = os.path.join('..', 'master_training_dataset.csv')
    master_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    create_master_dataset()
    