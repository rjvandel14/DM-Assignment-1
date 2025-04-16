import numpy as np
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import GRU, Dense #type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def train_xgboost(df):
    df = df.copy()
    feature_cols = [col for col in df.columns if col not in ['mood','id','date']]
    X = df[feature_cols]
    y = df['mood']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ XGBoost - MSE: {mse:.3f}, MAE: {mae:.3f}")

def train_gru(df):
    df = df.copy()
    df = df.sort_values(by=['id', 'date'])

    feature_cols = [col for col in df.columns if col not in ['mood','id','date']]
    sequence_length = 5
    x_seq, y_seq = [], []

    for user_id, group in df.groupby('id'):
        group = group.reset_index(drop=True)
        if len(group) < sequence_length + 1:
            continue
        for i in range(len(group) - sequence_length):
            seq = group.loc[i:i+sequence_length-1, feature_cols].values
            target = group.loc[i+sequence_length, 'mood']
            x_seq.append(seq)
            y_seq.append(target)

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)

    x_train, x_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    def scale(seq):
        reshaped = seq.reshape(-1, seq.shape[2])
        scaled = scaler.fit_transform(reshaped)
        return scaled.reshape(seq.shape)

    x_train = scale(x_train)
    x_test = scale(x_test)

    model = Sequential()
    model.add(GRU(64, input_shape=(sequence_length, x_seq.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=0)

    y_pred = model.predict(x_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ GRU - MSE: {mse:.3f}, MAE: {mae:.3f}")

import pandas as pd

if __name__ == "__main__":
    print("🚀 Start regression on median dataset")
    df_median = pd.read_csv("reduced_median.csv")
    print(f"Using {df_median.shape[1] - 3} features for median dataset")  # minus id, date, mood
    train_xgboost(df_median)
    train_gru(df_median)

    print("\n🚀 Start regression on kalman dataset")
    df_kalman = pd.read_csv("reduced_kalman.csv")
    print(f"Using {df_kalman.shape[1] - 3} features for kalman dataset")
    train_xgboost(df_kalman)
    train_gru(df_kalman)
