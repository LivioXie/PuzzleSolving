import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
import matplotlib.pyplot as plt

# Constants
PUZZLE_SIZE = 3  # 3x3 puzzle
SEED_SIZE = 100
EPOCHS = 1000
BATCH_SIZE = 32

# Generate random 3x3 sliding puzzle configurations
def generate_puzzle_batch(batch_size):
    puzzles = []
    for _ in range(batch_size):
        puzzle = np.arange(PUZZLE_SIZE * PUZZLE_SIZE)
        np.random.shuffle(puzzle)
        puzzle = puzzle.reshape((PUZZLE_SIZE, PUZZLE_SIZE))
        puzzles.append(puzzle)
    return np.array(puzzles)

# Generator
def build_generator(seed_size):
    model = Sequential()
    model.add(Dense(128, input_dim=seed_size))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(PUZZLE_SIZE * PUZZLE_SIZE, activation='softmax'))  # Output is a probability distribution
    model.add(Reshape((PUZZLE_SIZE, PUZZLE_SIZE)))
    return model

# Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(PUZZLE_SIZE, PUZZLE_SIZE)))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(1, activation='sigmoid'))
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Build models
generator = build_generator(SEED_SIZE)
discriminator = build_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)

# Training step
@tf.function
def train_step(puzzles):
    noise = tf.random.normal([BATCH_SIZE, SEED_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_puzzles = generator(noise, training=True)

        real_output = discriminator(puzzles, training=True)
        fake_output = discriminator(generated_puzzles, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for puzzle_batch in dataset:
            gen_loss, disc_loss = train_step(puzzle_batch)

        # Print progress
        print(f"Epoch {epoch + 1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}")

        # Generate and save a sample puzzle
        if (epoch + 1) % 100 == 0:
            sample_puzzle(generator)

# Function to sample a generated puzzle and plot
def sample_puzzle(generator_model):
    noise = tf.random.normal([1, SEED_SIZE])
    generated_puzzle = generator_model(noise, training=False).numpy().reshape((PUZZLE_SIZE, PUZZLE_SIZE))

    # Convert the generated puzzle to a valid permutation of numbers 0 to 8
    generated_puzzle = np.argsort(generated_puzzle.flatten())  # Ensure it's a valid permutation
    generated_puzzle = generated_puzzle.reshape((PUZZLE_SIZE, PUZZLE_SIZE))

    print("Generated Puzzle:")
    print(generated_puzzle)

    # Plot with labels
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(PUZZLE_SIZE):
        for j in range(PUZZLE_SIZE):
            num = int(generated_puzzle[i, j])  # Convert to int to display as a number
            color = 'lightblue' if num != 0 else 'white'  # Set color for blank space (0)
            ax.add_patch(plt.Rectangle((j, PUZZLE_SIZE-i-1), 1, 1, facecolor=color, edgecolor='black'))

            if num != 0:  # Only add number to non-zero cells
                ax.text(j + 0.5, PUZZLE_SIZE-i-0.5, str(num), color='black', ha='center', va='center', fontsize=20)

    ax.set_xlim(0, PUZZLE_SIZE)
    ax.set_ylim(0, PUZZLE_SIZE)
    ax.set_xticks(np.arange(0, PUZZLE_SIZE, 1))
    ax.set_yticks(np.arange(0, PUZZLE_SIZE, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.show()

# Generate dataset and train
puzzle_data = generate_puzzle_batch(1000)
puzzle_dataset = tf.data.Dataset.from_tensor_slices(puzzle_data).batch(BATCH_SIZE)

# Add after the training loop
# Save the trained models
def save_models():
    generator.save('c:/Users/livio/OneDrive/Documents/PuzzleSolving/PuzzleSolving/generator_model.keras')
    discriminator.save('c:/Users/livio/OneDrive/Documents/PuzzleSolving/PuzzleSolving/discriminator_model.keras')
    print("Models saved successfully!")

train(puzzle_dataset, EPOCHS)
save_models()  # Save models after training