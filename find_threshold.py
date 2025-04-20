import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def discretize_mood(score):
    if score < 6.5:
        return 0
    elif score < 7.5:
        return 1
    else:
        return 2

def find_best_features_and_save(df, name, thresholds=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], verbose=True, save_if_changed=True):
    df = df.copy()
    df['mood_class'] = df['mood'].apply(discretize_mood)

    original_features = [col for col in df.columns if col not in ['id', 'date', 'mood', 'mood_class']]
    x_full = df[original_features]
    y = df['mood_class']

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(x_full, y)
    importances = rf.feature_importances_

    best_score = 0
    best_features = []
    best_thresh = None

    for t in thresholds:
        selected = [f for f, imp in zip(original_features, importances) if imp > t]
        if not selected:
            continue

        x = df[selected]
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        score = f1_score(y_test, preds, average='macro')
        if verbose:
            print(f"Threshold: {t:.3f} | Features: {len(selected)} | F1 Score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_features = selected
            best_thresh = t

    if verbose:
        print(f"\n Best threshold: {best_thresh} with {len(best_features)} features")
        for feat in best_features:
            print(f"- {feat}")

    reduced_df = df[['id', 'date', 'mood'] + best_features].copy()

    if save_if_changed and set(best_features) != set(original_features):
        output_file = f"reduced_{name.lower()}.csv"
        reduced_df.to_csv(output_file, index=False)
        if verbose:
            print(f"Saved reduced dataset to: {output_file}")

    return reduced_df, best_features

# Load and run
df_median = pd.read_csv("all_df_median.csv")
reduced_median, features_median = find_best_features_and_save(df_median, name="median")

print("\n--- Median: Selected Features ---")
print(features_median)
print(f"Number of features kept: {len(features_median)}")
print(f"Reduced dataset shape: {reduced_median.shape}")

df_kalman = pd.read_csv("all_df_kalman.csv")
reduced_kalman, features_kalman = find_best_features_and_save(df_kalman, name="kalman")

print("\n--- Kalman: Selected Features ---")
print(features_kalman)
print(f"Number of features kept: {len(features_kalman)}")
print(f"Reduced dataset shape: {reduced_kalman.shape}")


