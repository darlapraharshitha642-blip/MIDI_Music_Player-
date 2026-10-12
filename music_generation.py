import numpy as np
import struct
import random

# ------------------------------------------
# 1. TRAINING DATA
# ------------------------------------------

# MIDI note numbers used for training
notes = [
    60, 62, 64, 65, 67, 67, 65, 64,
    62, 60, 60, 62, 64, 65, 67, 69,
    67, 65, 64, 62, 60, 62, 64, 60
]

# Convert notes into numerical indexes
unique_notes = sorted(set(notes))

note_to_index = {
    note: i for i, note in enumerate(unique_notes)
}

index_to_note = {
    i: note for note, i in note_to_index.items()
}

vocab_size = len(unique_notes)


# ------------------------------------------
# 2. CREATE INPUT AND TARGET SEQUENCES
# ------------------------------------------

X = []
Y = []

for i in range(len(notes) - 1):
    X.append(note_to_index[notes[i]])
    Y.append(note_to_index[notes[i + 1]])

X = np.array(X)
Y = np.array(Y)


# ------------------------------------------
# 3. SIMPLE RNN MODEL
# ------------------------------------------

hidden_size = 32
learning_rate = 0.05

rng = np.random.default_rng(1)

Wxh = rng.normal(0, 0.1, (hidden_size, vocab_size))
Whh = rng.normal(0, 0.1, (hidden_size, hidden_size))
Why = rng.normal(0, 0.1, (vocab_size, hidden_size))

bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# ------------------------------------------
# 4. TRAIN THE RNN
# ------------------------------------------

print("Training music generation model...")

for epoch in range(300):

    h = np.zeros((hidden_size, 1))
    hs = {-1: h}
    ps = {}
    loss = 0

    # Forward pass
    for t in range(len(X)):

        x = np.zeros((vocab_size, 1))
        x[X[t]] = 1

        h = np.tanh(
            np.dot(Wxh, x)
            + np.dot(Whh, h)
            + bh
        )

        y = np.dot(Why, h) + by
        p = softmax(y)

        ps[t] = p
        hs[t] = h

        loss += -np.log(p[Y[t], 0] + 1e-10)

    # Initialize gradients
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)

    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    dh_next = np.zeros_like(h)

    # Backpropagation through time
    for t in reversed(range(len(X))):

        dy = ps[t].copy()
        dy[Y[t]] -= 1

        dWhy += np.dot(dy, hs[t].T)
        dby += dy

        dh = np.dot(Why.T, dy) + dh_next

        dh_raw = (1 - hs[t] * hs[t]) * dh

        dbh += dh_raw

        x = np.zeros((vocab_size, 1))
        x[X[t]] = 1

        dWxh += np.dot(dh_raw, x.T)

        previous_h = hs[t - 1]
        dWhh += np.dot(dh_raw, previous_h.T)

        dh_next = np.dot(Whh.T, dh_raw)

    # Gradient clipping
    for gradient in [
        dWxh, dWhh, dWhy, dbh, dby
    ]:
        np.clip(gradient, -5, 5, out=gradient)

    # Update weights
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy

    bh -= learning_rate * dbh
    by -= learning_rate * dby

    if (epoch + 1) % 50 == 0:
        print(
            "Epoch:", epoch + 1,
            "Loss:", round(loss, 2)
        )


# ------------------------------------------
# 5. GENERATE NEW MUSIC
# ------------------------------------------

print("\nGenerating new music...")

generated_notes = []

current_index = note_to_index[notes[0]]
h = np.zeros((hidden_size, 1))

for _ in range(40):

    x = np.zeros((vocab_size, 1))
    x[current_index] = 1

    h = np.tanh(
        np.dot(Wxh, x)
        + np.dot(Whh, h)
        + bh
    )

    y = np.dot(Why, h) + by
    probabilities = softmax(y)

    # Select the next note
    current_index = random.choices(
        range(vocab_size),
        weights=probabilities.flatten(),
        k=1
    )[0]

    generated_notes.append(
        index_to_note[current_index]
    )

print("Generated notes:")
print(generated_notes)


# ------------------------------------------
# 6. MIDI FILE CREATION
# ------------------------------------------

def variable_length(value):

    buffer = value & 0x7F

    while value > 0x7F:
        value >>= 7
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80

    result = bytearray()

    while True:
        result.append(buffer & 0xFF)

        if buffer & 0x80:
            buffer >>= 8
        else:
            break

    return bytes(result)


def create_midi(note_list, filename):

    track = bytearray()

    # Set tempo to 120 BPM
    track.extend(
        b'\x00\xFF\x51\x03\x07\xA1\x20'
    )

    # Select piano instrument
    track.extend(b'\x00\xC0\x00')

    # Add each generated note
    for note in note_list:

        # Note ON
        track.extend(
            b'\x00' + bytes([0x90, note, 90])
        )

        # Note OFF after 480 ticks
        track.extend(
            variable_length(480)
            + bytes([0x80, note, 0])
        )

    # End of track
    track.extend(b'\x00\xFF\x2F\x00')

    # MIDI header
    header = (
        b'MThd'
        + struct.pack('>IHHH', 6, 0, 1, 480)
    )

    # MIDI track header
    track_header = (
        b'MTrk'
        + struct.pack('>I', len(track))
    )

    # Save the MIDI file
    with open(filename, 'wb') as file:
        file.write(header)
        file.write(track_header)
        file.write(track)


# ------------------------------------------
# 7. SAVE GENERATED MUSIC
# ------------------------------------------

create_midi(
    generated_notes,
    "generated_music.mid"
)

print("\nMusic generation completed!")
print("File created: generated_music.mid")
