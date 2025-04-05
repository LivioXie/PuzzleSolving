import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_discriminator(input_shape=(64, 64, 3), depth=64, dropout_rate=0.4):
    """
    Builds a discriminator model for image classification or GAN applications.
    
    Args:
        input_shape: Shape of the input images (height, width, channels)
        depth: Base depth for convolutional filters
        dropout_rate: Rate for dropout layers
        
    Returns:
        A compiled Keras model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(depth, kernel_size=4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Second convolutional block
    x = layers.Conv2D(depth * 2, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Third convolutional block
    x = layers.Conv2D(depth * 4, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Fourth convolutional block
    x = layers.Conv2D(depth * 8, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Flatten and output
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Create and save the model
    discriminator = build_discriminator()
    discriminator.summary()
    
    # Save the model
    save_path = r"c:\Users\livio\OneDrive\Documents\PuzzleSolving\PuzzleSolving\discriminator_model.keras"
    discriminator.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Example of how to load the model
    print("\nExample of loading the model:")
    print("loaded_model = keras.models.load_model('discriminator_model.keras')")