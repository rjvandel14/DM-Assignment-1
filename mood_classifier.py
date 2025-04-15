import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

def discretize_mood(score):
    if score <= 3:
        return 0  # Low
    elif score <= 6:
        return 1  # Medium
    else:
        return 2  # High

def train_random_forest(df):
    print("=== Training Random Forest ===")

    df = df.copy()
    df = df[df['imputed_flag'] == 0]  # only use clean data
    df['mood_class'] = df['mood_next_day'].apply(discretize_mood)

    feature_cols = [
        'mood_trend', 'activity_median', 'screen_trend',
        'call_sum', 'sms_sum', 'apps_trend',
        'arousal_trend', 'valence_trend'
    ]

    X = df[feature_cols]
    y = df['mood_class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1_macro'
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def train_lstm(df):
    print("=== Training LSTM ===")

    df = df.copy()
    df = df[df['imputed_flag'] == 0]
    df = df.sort_values(by=['user_id', 'date'])

    df['mood_class'] = df['mood_next_day'].apply(discretize_mood)

    features = ['mood', 'activity', 'screen', 'apps', 'call', 'sms', 'arousal', 'valence']
    sequence_length = 5
    x_seq, y_seq = [], []

    for user_id, group in df.groupby('user_id'):
        group = group.reset_index(drop=True)
        if len(group) < sequence_length + 1:
            continue
        for i in range(len(group) - sequence_length):
            seq = group.loc[i:i+sequence_length-1, features].values
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
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_labels))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))


# === Example Usage ===
if __name__ == "__main__":
    df = pd.read_csv("aggregated_mood_dataset.csv")
    train_random_forest(df)
    train_lstm(df)
