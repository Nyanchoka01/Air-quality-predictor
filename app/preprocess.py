import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(data_file):
    data = pd.read_excel(data_file)

    # Encode the target variable
    label_encoder = LabelEncoder()
    data['AQI_Class'] = label_encoder.fit_transform(data['AQI_Class'])

    X = data[['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']].values
    y = data['AQI_Class'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    width, height, channels = 1, 6, 1  # Assuming 1D CNN
    X_reshaped = X_scaled.reshape(-1, width, height, channels)

    return X_reshaped, y, label_encoder
