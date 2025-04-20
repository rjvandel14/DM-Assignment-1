import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import GRU, Dense #type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

def plot_pred_vs_actual(y_test, y_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([1, 10], [1, 10], '--', color='gray')  # 45-degree line
    plt.xlabel("True Mood")
    plt.ylabel("Predicted Mood")
    plt.title(f"Predicted vs Actual Mood ({title})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_xgboost(df):
    df = df.copy()
    feature_cols = [col for col in df.columns if col not in ['mood','id','date']]
    x = df[feature_cols]
    y = df['mood']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ XGBoost - MSE: {mse:.3f}, MAE: {mae:.3f}")
    plot_pred_vs_actual(y_test, y_pred, "XGBoost")

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

    y_pred = model.predict(x_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"GRU - MSE: {mse:.3f}, MAE: {mae:.3f}")
    plot_pred_vs_actual(y_test, y_pred, "GRU")


if __name__ == "__main__":
    print("Start regression on median dataset")
    df_median = pd.read_csv("reduced_median.csv")
    print(f"Using {df_median.shape[1] - 3} features for median dataset")  # minus id, date, mood
    train_xgboost(df_median)
    train_gru(df_median)

    print("\n Start regression on kalman dataset")
    df_kalman = pd.read_csv("reduced_kalman.csv")
    print(f"Using {df_kalman.shape[1] - 3} features for kalman dataset")
    train_xgboost(df_kalman)
    train_gru(df_kalman)
