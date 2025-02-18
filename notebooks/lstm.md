Data Preprocessing
Sorting and Grouping Events by User
First, you’ll want to group events by user (or session) and order them by time. This gives you the sequential data needed for training a sequence model.
Copy
Edit
import pandas as pd

# Example: Load one chunk of your dataset (or a consolidated DataFrame)
df = pd.read_csv('../datasets/2025_csv/_chunk_0_100000.csv', nrows=10000)

# Convert the event_time to datetime
df['event_time'] = pd.to_datetime(df['event_time'])

# Sort the DataFrame by user_id and event_time
df.sort_values(by=['user_id', 'event_time'], inplace=True)

# Group by user_id and collect the event types as sequences
user_sequences = df.groupby('user_id')['event_properties_action'].apply(list)
Note: In this example, I used event_properties_action as the “action” indicator. Depending on your objectives, you might want to use or combine other columns (e.g., event_type, event_properties_displayName) as the event label.

b. Tokenizing/Coding the Events
Since our model needs numerical inputs, map each unique event (or a combination of features) to an integer token.

python
Copy
Edit
from sklearn.preprocessing import LabelEncoder

# Flatten the list of events to fit the encoder
all_events = df['event_properties_action'].astype(str).tolist()
encoder = LabelEncoder()
encoder.fit(all_events)

# Map each event in user_sequences to its integer representation
user_sequences_encoded = user_sequences.apply(lambda seq: encoder.transform(seq).tolist())
Now you have sequences like [3, 5, 2, 7, ...] where each number represents a specific event.

2. Preparing Data for Sequence Modeling
a. Creating Input-Output Pairs
For sequence models, you typically use a sliding window approach. For example, given a sequence of events 
[
𝑒
1
,
𝑒
2
,
.
.
.
,
𝑒
𝑛
]
[e 
1
​
 ,e 
2
​
 ,...,e 
n
​
 ], use the first 
𝑛
−
1
n−1 events as input and predict the 
𝑛
𝑡
ℎ
n 
th
  event.

python
Copy
Edit
import numpy as np

# Define a function to create input-output pairs from sequences
def create_sequences(sequence, window_size=3):
    X, y = [], []
    for i in range(len(sequence) - window_size):
        X.append(sequence[i:i+window_size])
        y.append(sequence[i+window_size])
    return np.array(X), np.array(y)

# For demonstration, let's use a fixed window size of 3
X_list, y_list = [], []
for seq in user_sequences_encoded:
    if len(seq) > 3:
        X_seq, y_seq = create_sequences(seq, window_size=3)
        X_list.append(X_seq)
        y_list.append(y_seq)

# Concatenate all user sequences together
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
b. Data Splitting
Split the data into training and validation sets.

python
Copy
Edit
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
3. Building a Sequence-Based Model
a. Model with LSTM
You can now build a simple LSTM model using Keras/TensorFlow.

python
Copy
Edit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Define some parameters
vocab_size = len(encoder.classes_)  # Total number of unique events
embedding_dim = 50                   # Embedding size (adjust as needed)
lstm_units = 64                      # Number of LSTM units

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=X.shape[1]),
    LSTM(lstm_units),
    Dense(vocab_size, activation='softmax')  # Softmax for multi-class classification
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
b. Training the Model
python
Copy
Edit
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))
4. Making Predictions
Once trained, you can predict the next action given a sequence of events. For example:

python
Copy
Edit
def predict_next_action(sequence, window_size=3):
    # Ensure sequence is encoded and padded to the required window size
    sequence = encoder.transform(sequence[-window_size:])  # Use the last `window_size` events
    sequence = np.expand_dims(sequence, axis=0)  # Model expects batch dimension
    pred = model.predict(sequence)
    next_event_token = np.argmax(pred, axis=1)[0]
    next_event = encoder.inverse_transform([next_event_token])[0]
    return next_event

# Example usage:
sample_sequence = ['modelRequest', 'formSubmit', 'buttonClick']  # Replace with actual events
predicted_action = predict_next_action(sample_sequence)
print("Predicted next action:", predicted_action)
5. Next Steps & Considerations
Feature Engineering:
While we focused on a single event column (event_properties_action), consider using additional features (like event time differences, session information, or user properties) to enrich your model.

Handling Sequence Lengths:
Not all sequences will have the same length. You might need to pad sequences to a fixed length (e.g., using tf.keras.preprocessing.sequence.pad_sequences).

Model Tuning:
Experiment with different window sizes, LSTM units, or even try alternative models like GRUs or Transformer-based architectures if your dataset is large enough.

Evaluation Metrics:
Apart from accuracy, consider metrics like Top-K accuracy since recommending the top few actions might be beneficial in practice.

This approach gives you a clear pipeline from raw data to a trained sequence-based model that predicts the next action for a user. Let me know if you need further clarification on any of these steps or additional code details!