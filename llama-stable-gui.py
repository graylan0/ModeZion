import tkinter as tk
import threading
import os
import aiosqlite
import logging
import numpy as np
import base64
import queue
import uuid
import customtkinter
import requests
import io
import sys
import random
import asyncio
import weaviate
from concurrent.futures import ThreadPoolExecutor
from summa import summarizer
from textblob import TextBlob
from weaviate.util import generate_uuid5
from PIL import Image, ImageTk
from llama_cpp import Llama


# Create a FIFO queue
q = queue.Queue()
DB_NAME = "story_generator.db"
logger = logging.getLogger(__name__)

WEAVIATE_ENDPOINT = "https://"  # Replace with your Weaviate instance URL
WEAVIATE_QUERY_PATH = "/v1/graphql"

client = weaviate.Client(
    url="https://",
)

# Database initialization
async def init_db():
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            # Create Responses Table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trideque_point INT,
                    response TEXT,
                    response_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INT
                )
            """)

            # Create Context Table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trideque_point INT,
                    summarization_context TEXT,
                    full_text TEXT
                )
            """)

            # Create Users Table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    relationship_state TEXT
                )
            """)

            await db.commit()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


llm = Llama(
  model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
  n_gpu_layers=-1,
  n_ctx=3900,
)


def llama_generate(prompt, max_tokens=2500, chunk_size=500):
    try:
        # Function to dynamically determine overlap based on context
        def find_overlap(chunk, next_chunk):
            # Define the maximum possible overlap
            max_overlap = min(len(chunk), 100)  # e.g., 100 characters
            for overlap in range(max_overlap, 0, -1):
                if chunk.endswith(next_chunk[:overlap]):
                    return overlap
            return 0

        # Split the prompt into initial chunks without overlap
        prompt_chunks = [prompt[i:i+chunk_size] for i in range(0, len(prompt), chunk_size)]

        responses = []
        for i, chunk in enumerate(prompt_chunks):
            output = llm(chunk, max_tokens=min(max_tokens, chunk_size))
            responses.append(output)

            # If not the last chunk, dynamically determine the overlap with the next chunk
            if i < len(prompt_chunks) - 1:
                overlap = find_overlap(output, prompt_chunks[i + 1])
                prompt_chunks[i + 1] = output[-overlap:] + prompt_chunks[i + 1]

        # Concatenate responses
        final_response = ''.join(responses)

        # Optional: Additional post-processing for continuity and coherence
        # This could include checking for incomplete sentences, ensuring logical flow, etc.

        return final_response
    except Exception as e:
        logger.error(f"Error in llama_generate: {e}")
        return None  # or return an appropriate default value or message


def run_async_in_thread(loop, coro_func, *args):
    try:
        asyncio.set_event_loop(loop)
        coro = coro_func(*args)
        loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Error in async thread: {e}")
    finally:
        loop.close()
    
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.setup_gui()
        self.response_queue = queue.Queue()
        self.client = weaviate.Client(url="https://")
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as needed
        
    async def retrieve_past_interactions(self, theme, result_queue):
        try:
            def sync_query():
                query = {
                    "class": "InteractionHistory",
                        "properties": ["user_message", "ai_response"],
                    "where": {
                        "operator": "GreaterThan",
                        "path": ["certainty"],
                        "valueFloat": 0.7
                    }
                }
                return self.client.query.raw(query).do()

            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(executor, sync_query)

            if response.get('data', {}).get('InteractionHistory', []):
                interactions = response['data']['InteractionHistory']
                result_queue.put(interactions)
            else:
                logger.error("No interactions found for the given theme.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])




    def process_response_and_store_in_weaviate(self, user_message, ai_response):
        # Analyze the response using TextBlob
        response_blob = TextBlob(ai_response)
        keywords = response_blob.noun_phrases  # Extracting noun phrases as keywords
        sentiment = response_blob.sentiment.polarity  # Sentiment analysis

        # Map the response to Weaviate schema
        interaction_object = {
            "userMessage": user_message,
            "aiResponse": ai_response,
            "keywords": list(keywords),
            "sentiment": sentiment
        }

        # Generate a UUID for the new object
        interaction_uuid = str(uuid.uuid4())

        # Store the object in Weaviate
        try:
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=interaction_uuid
            )
            print(f"Interaction stored in Weaviate with UUID: {interaction_uuid}")
        except Exception as e:
            print(f"Error storing interaction in Weaviate: {e}")

    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown(wait=True)

    def create_interaction_history_object(self, user_message, ai_response):
        interaction_object = {
            "user_message": user_message,
            "ai_response": ai_response
        }

        try:
            # Generate a UUID for the new object
            object_uuid = uuid.uuid4()
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=object_uuid
            )
            print(f"Interaction history object created with UUID: {object_uuid}")
        except Exception as e:
            print(f"Error creating interaction history object in Weaviate: {e}")

    def map_keywords_to_weaviate_classes(self, keywords, context):
        try:
            # Attempt to summarize the context using SUMMA
            summarized_context = summarizer.summarize(context)
        except Exception as e:
            print(f"Error in summarizing context: {e}")
            summarized_context = context  # Fallback to original context if summarization fails

        try:
            # Attempt to analyze the sentiment of the summarized context using TextBlob
            sentiment = TextBlob(summarized_context).sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            sentiment = TextBlob("").sentiment  # Fallback to neutral sentiment

        # Define class mappings based on sentiment and context
        positive_class_mappings = {
            "keyword1": "PositiveClassA",
                "keyword2": "PositiveClassB",
            # Add more mappings for positive sentiment
        }

        negative_class_mappings = {
            "keyword1": "NegativeClassA",
            "keyword2": "NegativeClassB",
            # Add more mappings for negative sentiment
        }

        # Default mapping if no specific sentiment-based mapping is found
        default_mapping = {
            "keyword1": "NeutralClassA",
            "keyword2": "NeutralClassB",
            # Add more default mappings
        }

        # Determine which mapping to use based on sentiment
        if sentiment.polarity > 0:
            mapping = positive_class_mappings
        elif sentiment.polarity < 0:
            mapping = negative_class_mappings
        else:
            mapping = default_mapping

        # Map keywords to classes with error handling
        mapped_classes = {}
        for keyword in keywords:
            try:
                if keyword in mapping:
                    mapped_classes[keyword] = mapping[keyword]
            except KeyError as e:
                print(f"Error in mapping keyword '{keyword}': {e}")

        return mapped_classes


    def run_async_in_thread(loop, coro_func, message, result_queue):
        asyncio.set_event_loop(loop)
        coro = coro_func(message, result_queue)  # Create the coroutine here
        loop.run_until_complete(coro)

    def generate_response(self, message):
        try:
            result_queue = queue.Queue()
            loop = asyncio.new_event_loop()
            past_interactions_thread = threading.Thread(target=run_async_in_thread, args=(loop, self.retrieve_past_interactions, message, result_queue))
            past_interactions_thread.start()
            past_interactions_thread.join()

            past_interactions = result_queue.get()

            past_context = "\n".join([f"User: {interaction['user_message']}\nAI: {interaction['ai_response']}" for interaction in past_interactions])
            complete_prompt = f"{past_context}\nUser: {message}"

            response = llama_generate(complete_prompt)
            response_text = response['choices'][0]['text']
            self.response_queue.put({'type': 'text', 'data': response_text})

            context = self.retrieve_context(message)
            keywords = self.extract_keywords(message)
            mapped_classes = self.map_keywords_to_weaviate_classes(keywords, context)

            self.create_interaction_history_object(message, response_text)

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")

    def on_submit(self, event=None):
        message = self.entry.get().strip()
        if message:
            self.entry.delete(0, tk.END)
            self.text_box.insert(tk.END, f"You: {message}\n")
            self.text_box.see(tk.END)
            self.executor.submit(self.generate_response, message)
            self.executor.submit(self.generate_images, message)
            self.after(100, self.process_queue)

    def create_object(self, class_name, object_data):
        # Generate a unique string for the object
        unique_string = f"{object_data['time']}-{object_data['user_message']}-{object_data['ai_response']}"

        # Generate a UUID based on the unique string using a predefined namespace
        object_uuid = uuid.uuid5(uuid.NAMESPACE_URL, unique_string).hex

        # Insert the object into Weaviate
        try:
            self.client.data_object.create(object_data, object_uuid, class_name)
            print(f"Object created with UUID: {object_uuid}")
        except Exception as e:
            print(f"Error creating object in Weaviate: {e}")

        return object_uuid

    def process_queue(self):
        try:
            while True:
                response = self.response_queue.get_nowait()
                if response['type'] == 'text':
                    self.text_box.insert(tk.END, f"AI: {response['data']}\n")
                elif response['type'] == 'image':
                    self.image_label.config(image=response['data'])
                    self.image_label.image = response['data']  # keep a reference to the image
                self.text_box.see(tk.END)
        except queue.Empty:
            self.after(100, self.process_queue)
            
    def extract_keywords(self, message):
        blob = TextBlob(message)
        nouns = blob.noun_phrases
        return list(nouns)

    async def retrieve_past_interactions(self, theme, result_queue):
        try:
            # Define a function to perform the synchronous part
            def sync_query():
                return self.client.query.get("interaction_history", ["user_message", "ai_response"]).with_near_text({
                    "concepts": [theme],
                    "certainty": 0.7
                }).do()

            # Run the synchronous function in a separate thread
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(executor, sync_query)

            # Check and process the response
            if 'data' in response and 'Get' in response['data'] and 'interaction_history' in response['data']['Get']:
                interactions = response['data']['Get']['interaction_history']
                result_queue.put(interactions)
            else:
                logger.error("No interactions found for the given theme.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])


     
    def generate_images(self, message):
        try:
            url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
            payload = {
                "prompt": message,
                "steps" : 50,
                "seed" : random.randrange(sys.maxsize),
                "enable_hr": "false",
                "denoising_strength": "0.7",
                "cfg_scale" : "7",
                "width": 1280,
                "height": 512,
                "restore_faces": "true",
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                try:
                    r = response.json()
                    for i in r['images']:
                        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
                        img_tk = ImageTk.PhotoImage(image)
                        self.response_queue.put({'type': 'image', 'data': img_tk})
                        self.image_label.image = img_tk  # keep a reference to the image
                except ValueError as e:
                    print("Error processing image data: ", e)
            else:
                print("Error generating image: ", response.status_code)

        except Exception as e:
             logger.error(f"Error in generate_images: {e}")

    def setup_gui(self):
        # Configure window
        self.title("OneLoveIPFS AI")
        self.geometry(f"{1100}x{580}")

        # Configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Load logo image and display in sidebar frame
        logo_path = os.path.join(os.getcwd(), "logo.png")
        logo_img = Image.open(logo_path).resize((140, 77))  # Add the .resize() method with the desired dimensions
        logo_photo = ImageTk.PhotoImage(logo_img)  # Convert PIL.Image to tkinter.PhotoImage
        self.logo_label = tk.Label(self.sidebar_frame, image=logo_photo, bg=self.sidebar_frame["bg"])  # Create a tkinter.Label
        self.logo_label.image = logo_photo  # Keep a reference to the image
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))  # This is the correct position for the logo_label grid statement

        # Create text box
        self.text_box = customtkinter.CTkTextbox(self, bg_color="white", text_color="white", border_width=0, height=20, width=50, font=customtkinter.CTkFont(size=13))
        self.text_box.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="Chat With Llama")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.on_submit)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")

        self.entry.bind('<Return>', self.on_submit)

        # Create a label to display the image
        self.image_label = tk.Label(self)
        self.image_label.grid(row=4, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

if __name__ == "__main__":
    try:
        app = App()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(init_db())
        app.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")
