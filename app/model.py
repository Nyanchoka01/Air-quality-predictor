# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model

def build_model():
    input_shape = (6, 1, 1)  # Shape of input data

    model_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 1), activation='relu')(model_input)
    x = MaxPooling2D((2, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 1))(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)  # 3 classes: Class_A, Class_B, Class_C

    model = Model(inputs=model_input, outputs=output)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

