import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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
    
def evaluate_baseline(x_train, y_train, x_test, y_test):
    print("\n=== Baseline (Majority Class) ===")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    y_dummy = dummy.predict(x_test)

    print(classification_report(y_test, y_dummy))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_dummy))

def train_random_forest(df):
    print("=== Training Random Forest ===")

    df = df.copy()
    df['mood_class'] = df['mood'].apply(discretize_mood)

    feature_cols = [col for col in df.columns if col not in ['mood','id','date','mood_class']]

    x = df[feature_cols]
    y = df['mood_class']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        'n_estimators': [100], # Number of trees in the forest
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        cv=5,
        scoring='f1_macro'
    )
    grid_search.fit(x_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(x_test)

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    evaluate_baseline(x_train, y_train, x_test, y_test)

    plt.figure(figsize=(10, 5))
    importances = best_rf.feature_importances_
    sns.barplot(x=importances, y=feature_cols)
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Final RF performance: Accuracy = {acc:.3f} | Macro F1 = {f1:.3f}")

def train_lstm(df):
    print("=== Training LSTM ===")

    df = df.copy()
    df = df.sort_values(by=['id', 'date'])

    df['mood_class'] = df['mood'].apply(discretize_mood)

    feature_cols = [col for col in df.columns if col not in ['mood','id','date','mood_class']]
    sequence_length = 5
    x_seq, y_seq = [], []

    for user_id, group in df.groupby('id'):
        group = group.reset_index(drop=True)
        if len(group) < sequence_length + 1:
            continue
        for i in range(len(group) - sequence_length):
            seq = group.loc[i:i+sequence_length-1, feature_cols].values
            target = group.loc[i+sequence_length, 'mood_class']
            x_seq.append(seq)
            y_seq.append(target)

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)

    x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    num_features = x_seq.shape[2]

    def scale(seq):
        seq_2d = seq.reshape(-1, num_features)
        scaled = scaler.fit_transform(seq_2d)
        return scaled.reshape(seq.shape)

    x_train = scale(x_train)
    x_val = scale(x_val)
    x_test = scale(x_test)

    # Build model
    model = Sequential()
    model.add(LSTM(64, input_shape=(sequence_length, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32, callbacks=[early_stop], verbose=0)

    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

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

    acc = accuracy_score(y_test, y_pred_labels)
    f1 = f1_score(y_test, y_pred_labels, average='macro')
    print(f"Final LSTM performance: Accuracy = {acc:.3f} | Macro F1 = {f1:.3f}")

if __name__ == "__main__":
    print("Start classification median.")
    df_median = pd.read_csv("reduced_median.csv")
    print(f"Using {df_median.shape[1] - 3} features for median dataset")  # -3 for id, date, mood

    print_mood_distribution(df_median, "Median")
    train_random_forest(df_median)
    train_lstm(df_median)
    
    print("Start classification kalman.")
    df_kalman = pd.read_csv("reduced_kalman.csv")
    print(f"Using {df_kalman.shape[1] - 3} features for median dataset")  # -3 for id, date, mood

    print_mood_distribution(df_kalman, "Kalman")
    train_random_forest(df_kalman)
    train_lstm(df_kalman)
    
