import json
import sqlite3
import threading
import time
import openai
from tkinter import Tk, Text
from customtkinter import CTkButton, CTkEntry

class App(Tk):
    def __init__(self):
        super().__init__()

        # Load the configuration
        self.config = self.load_config()

        # Set the OpenAI API key
        openai.api_key = self.config['openai_api_key']

        # Set up the GUI
        self.title("AI Chat")
        self.geometry("800x600")

        # Create a text box for the chat history
        self.text_box = Text(self)
        self.text_box.pack(pady=10)

        # Create an entry box for user input
        self.entry = CTkEntry(self)
        self.entry.pack(pady=10)

        # Create a button to send the user's message
        self.button = CTkButton(self, text="Send", command=self.on_send)
        self.button.pack()

        # Initialize the memory
        self.memory = []

        # Connect to the database
        self.conn = sqlite3.connect('embeddings.db', check_same_thread=False)
        self.c = self.conn.cursor()

        # Create the table (if it doesn't exist)
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                embedding TEXT,
                timestamp INTEGER
            )
        ''')

    def load_config(self):
        with open('config.json') as config_file:
            config = json.load(config_file)
        return config

    def on_send(self):
        # Get the user's message
        message = self.entry.get()

        # Clear the entry box
        self.entry.delete(0, 'end')

        # Add the user's message to the chat history
        self.text_box.insert('end', f"You: {message}\n")

        # Generate a response from the AI
        threading.Thread(target=self.generate_response, args=(message,)).start()

    def get_embedding(self, text: str, model="text-embedding-ada-002") -> list[float]:
        return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

    def generate_response(self, message):
        # Add the user's message to the memory
        self.memory.append({
            'role': 'user',
            'content': message
        })

        # Retrieve the embeddings from the database
        self.c.execute('''
            SELECT embedding, timestamp
            FROM embeddings
        ''')
        embeddings_and_timestamps = self.c.fetchall()

        # Calculate the decay factor for each embedding based on its age
        current_time = int(time.time())
        decayed_embeddings = []
        for embedding, timestamp in embeddings_and_timestamps:
            age = current_time - timestamp
            decay_factor = 0.5 ** (age / (60 * 60 * 24))  # Half-life of one day
            decayed_embedding = [value * decay_factor for value in json.loads(embedding)]
            decayed_embeddings.append(decayed_embedding)

        # Use the decayed embeddings as input to the model
        average_embedding = [sum(values) / len(values) for values in zip(*decayed_embeddings)]

        # Calculate weights for each message based on the embeddings
        weights = self.calculate_weights(self.memory, average_embedding)

        # Select a subset of the messages based on the weights
        selected_messages = self.select_messages(self.memory, weights)

        # Call the chat model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=selected_messages
        )

        # Extract the assistant's message from the response
        response_text = response['choices'][0]['message']['content']

        # Add the assistant's message to the chat history
        self.text_box.insert('end', f"AI: {response_text}\n")

        # Add the assistant's message to the memory
        self.memory.append({
            'role': 'assistant',
            'content': response_text
        })

        # Get the assistant's embedding and store it in the database
        assistant_embedding = self.get_embedding(response_text)
        self.c.execute('''
            INSERT INTO embeddings (embedding, timestamp)
            VALUES (?, ?)
        ''', (json.dumps(assistant_embedding), int(time.time())))

        # Commit the changes to the database
        self.conn.commit()

    def calculate_weights(self, memory, average_embedding):
        weights = []
        for message in memory:
            message_embedding = self.get_embedding(message['content'])
            similarity = sum(a * b for a, b in zip(message_embedding, average_embedding))
            weights.append(similarity)
        return weights

    def select_messages(self, memory, weights):
        # Pair each message with its weight
        pairs = list(zip(memory, weights))

        # Sort the pairs by weight in descending order
        pairs.sort(key=lambda pair: pair[1], reverse=True)

        # Always include the last 2 messages
        selected_pairs = pairs[-2:]

        # Fill the rest with the messages with the highest weights
        selected_pairs += pairs[:3]

        # Extract the messages from the pairs
        selected_messages = [pair[0] for pair in selected_pairs]

        return selected_messages

if __name__ == "__main__":
    # Create and run the app
    app = App()
    app.mainloop()
