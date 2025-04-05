import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_generator(latent_dim=100, output_shape=(64, 64, 3), depth=64):
    """
    Builds a generator model for a GAN.
    
    Args:
        latent_dim: Dimension of the latent space input
        output_shape: Shape of the output images (height, width, channels)
        depth: Base depth for convolutional filters
        
    Returns:
        A Keras model
    """
    # Calculate initial size based on output shape
    initial_height = output_shape[0] // 16
    initial_width = output_shape[1] // 16
    initial_channels = depth * 8
    
    # Input layer
    inputs = keras.Input(shape=(latent_dim,))
    
    # Dense layer and reshape
    x = layers.Dense(initial_height * initial_width * initial_channels)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((initial_height, initial_width, initial_channels))(x)
    
    # First upsampling block
    x = layers.Conv2DTranspose(depth * 4, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Second upsampling block
    x = layers.Conv2DTranspose(depth * 2, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Third upsampling block
    x = layers.Conv2DTranspose(depth, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Final upsampling block
    x = layers.Conv2DTranspose(output_shape[2], kernel_size=4, strides=2, padding='same', activation='tanh')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=x, name="generator")
    
    return model

if __name__ == "__main__":
    # Create and save the model
    generator = build_generator()
    generator.summary()
    
    # Save the model
    save_path = r"c:\Users\livio\OneDrive\Documents\PuzzleSolving\PuzzleSolving\generator_model.keras"
    generator.save(save_path)
    print(f"Model saved to {save_path}")