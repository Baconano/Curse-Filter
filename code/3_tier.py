import pandas as pd
import os

def create_3_tier_dataset():
    print("Starting Strategic 3-Tier Merge...")

    # Paths relative to the 'code' folder
    jigsaw_path = os.path.join('..', 'datasets', 'train.csv')
    davidson_path = os.path.join('..', 'datasets', 'labeled_data.csv')

    # 1. Process Davidson
    if os.path.exists(davidson_path):
        dav = pd.read_csv(davidson_path)
        dav_clean = dav[['tweet', 'class']].copy()
        # FIX: Map BOTH 0 (Hate) and 1 (Offensive) to 2 (Mute/Tier 2)
        dav_clean['label'] = dav_clean['class'].map({0: 2, 1: 2, 2: 0})
        dav_clean = dav_clean[['tweet', 'label']]
        dav_clean.columns = ['text', 'label']
    else:
        print("Davidson file not found!")
        return

    # 2. Process Jigsaw
    if os.path.exists(jigsaw_path):
        jig = pd.read_csv(jigsaw_path)
        def jigsaw_map(row):
            if row['severe_toxic'] or row['identity_hate'] or row['threat']:
                return 2
            if row['toxic'] or row['insult'] or row['obscene']:
                return 1
            return 0
        jig['label'] = jig.apply(jigsaw_map, axis=1)
        jig_clean = jig[['comment_text', 'label']].copy()
        jig_clean.columns = ['text', 'label']
    else:
        print("Jigsaw file not found!")
        return

    # 3. STRATEGIC INJECTION (The Point Correction Fix)
    # This list ensures "fuck you" is Tier 2, while "fuck" and "you" are Tier 0
    targeted_toxic = ["fuck you", "you are a bitch", "you bitch", "shut up pussy", "get out pussy", "kill yourself", "fuck u"]
    slurs = ["bitch", "pussy", "faggot", "nigger", "cunt", "retard", "whore", "fucker", "nigga", "fag", "gay"]
    neutral_you = ["you", "are you", "how are you", "is that you", "thank you", "you are", "you there"]
    general_cursing = ["fuck", "damn", "shit", "hell", "ass", "fucking hell"]

    # Boost Slurs & Targeted Attacks to Tier 2 (MUTE)
    # We use 5000 copies to ensure the AI prioritizes these combinations
    tier2_data = [(t, 2) for t in targeted_toxic + slurs] * 5000 
    
    # Neutralize 'You' and 'General Swearing' to Tier 0 (CLEAN)
    tier0_data = [(n, 0) for n in neutral_you + general_cursing] * 5000

    custom_df = pd.DataFrame(tier2_data + tier0_data, columns=['text', 'label'])

    # 4. Final Merge
    master_3tier = pd.concat([dav_clean, jig_clean, custom_df], ignore_index=True)
    master_3tier = master_3tier.dropna(subset=['text'])
    
    output_path = os.path.join('..', 'datasets', '3tier_training_data.csv')
    master_3tier.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"Success! Master Dataset created at {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    create_3_tier_dataset()