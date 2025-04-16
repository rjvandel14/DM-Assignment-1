import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import os

def discretize_mood(score):
    if score < 6.5:
        return 0  # Low
    elif score < 7.5:
        return 1  # Medium
    else:
        return 2  # High

def print_mood_distribution(df, name=""):
    low = df[df['mood'] < 6.5].shape[0]
    medium = df[(df['mood'] >= 6.5) & (df['mood'] < 7.5)].shape[0]
    high = df[df['mood'] >= 7.5].shape[0]
    print(f"\n=== Mood distribution ({name}) ===")
    print(f"Low: {low}  |  Medium: {medium}  |  High: {high}")
    print(f"Total: {low + medium + high} (sanity: {len(df)})")

def evaluate_baseline(X_train, y_train, X_test, y_test):
    print("\n=== Baseline (Majority Class) ===")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)

    print(classification_report(y_test, y_dummy))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_dummy))

def find_best_features(df, target_col, thresholds):
    df['mood_class'] = df['mood'].apply(discretize_mood)
    feature_cols = [col for col in df.columns if col not in ['mood', 'id', 'date', 'mood_class']]

    rf_temp = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_temp.fit(df[feature_cols], df['mood_class'])
    importances = rf_temp.feature_importances_

    best_score = 0
    best_features = []
    best_thresh = None

    for thresh in thresholds:
        selected_features = [f for f, imp in zip(feature_cols, importances) if imp > thresh]
        if not selected_features:
            continue

        X = df[selected_features]
        y = df['mood_class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Threshold: {thresh:.3f} | Features: {len(selected_features)} | F1 Score: {f1:.3f}")
        if f1 > best_score:
            best_score = f1
            best_features = selected_features
            best_thresh = thresh

    print(f"\n✅ Best threshold: {best_thresh:.3f} with {len(best_features)} features")
    print("Selected features:")
    for feat in best_features:
        print(f"- {feat}")

    return best_features

def train_random_forest(df, best_features):
    print("=== Training Random Forest ===")

    df = df.copy()
    df['mood_class'] = df['mood'].apply(discretize_mood)

    X = df[best_features]
    y = df['mood_class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)

    print("\n=== Final Random Forest Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    evaluate_baseline(X_train, y_train, X_test, y_test)

    plt.figure(figsize=(10, 5))
    importances_final = best_rf.feature_importances_
    sns.barplot(x=importances_final, y=best_features)
    plt.title("Random Forest Feature Importances (Final)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def train_lstm(df, best_features):
    print("=== Training LSTM ===")

    df = df.copy()
    df = df.sort_values(by=['id', 'date'])
    df['mood_class'] = df['mood'].apply(discretize_mood)

    sequence_length = 5
    x_seq, y_seq = [], []

    for user_id, group in df.groupby('id'):
        group = group.reset_index(drop=True)
        if len(group) < sequence_length + 1:
            continue
        for i in range(len(group) - sequence_length):
            seq = group.loc[i:i+sequence_length-1, best_features].values
            target = group.loc[i+sequence_length, 'mood_class']
            x_seq.append(seq)
            y_seq.append(target)

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)

    x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    scaler = StandardScaler()
    num_features = x_seq.shape[2]

    def scale(seq):
        seq_2d = seq.reshape(-1, num_features)
        scaled = scaler.fit_transform(seq_2d)
        return scaled.reshape(seq.shape)

    x_train = scale(x_train)
    x_val = scale(x_val)
    x_test = scale(x_test)

    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32, callbacks=[early_stop], verbose=0)

    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print("\n=== LSTM Evaluation ===")
    print(classification_report(y_test, y_pred_labels))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("LSTM Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for name, path in zip(["Median", "Kalman"], ["all_df_median.csv", "all_df_kalman.csv"]):
        print(f"Start classification {name.lower()}.")
        df = pd.read_csv(path)
        df['mood_class'] = df['mood'].apply(discretize_mood)

        best_features = find_best_features(df, 'mood_class', thresholds=[0, 0.01, 0.02, 0.03, 0.04, 0.05])

        drop_cols = [col for col in df.columns if col not in ['id', 'date', 'mood', 'mood_class'] + best_features]
        df = df.drop(columns=drop_cols, errors='ignore')

        print_mood_distribution(df, name)

        reduced_df = df[['id', 'date', 'mood'] + best_features].copy()
        output_path = f"reduced_{name.lower()}.csv"
        reduced_df.to_csv(output_path, index=False)
        print(f"Saved reduced dataset to {output_path}")

        train_random_forest(df, best_features)
        train_lstm(df, best_features)
