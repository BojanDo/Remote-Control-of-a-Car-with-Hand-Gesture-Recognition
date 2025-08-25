import tensorflow as tf
from config import SAMPLES_PER_SEQUENCE, GESTURES


'''
#GOOD
def build_model():
    regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)

    model = tf.keras.Sequential([
        tf.keras.layers.Input((SAMPLES_PER_SEQUENCE, 6), batch_size=1),
        tf.keras.layers.Conv1D(16, kernel_size=7, strides=2, activation='relu',
                               padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.Conv1D(8, kernel_size=5, strides=2, activation='relu',
                               padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation='relu',
                              kernel_regularizer=regularizer, bias_regularizer=regularizer,
                              activity_regularizer=regularizer),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(GESTURES), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    return model'''


#BETTER
def build_model():
    regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)

    model = tf.keras.Sequential([
        tf.keras.layers.Input((SAMPLES_PER_SEQUENCE, 6), batch_size=1),

        tf.keras.layers.Conv1D(16, kernel_size=7, strides=2, padding='same',
                               kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Conv1D(12, kernel_size=5, strides=2, padding='same',
                               kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Conv1D(8, kernel_size=3, strides=1, padding='same',
                               kernel_regularizer=regularizer, bias_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(16, activation='relu',
                              kernel_regularizer=regularizer, bias_regularizer=regularizer,
                              activity_regularizer=regularizer),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(len(GESTURES), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model


