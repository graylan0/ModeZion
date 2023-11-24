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


q = queue.Queue()
DB_NAME = "story_generator.db"
logger = logging.getLogger(__name__)

WEAVIATE_ENDPOINT = "https://"  # Replace with your Weaviate instance URL
WEAVIATE_QUERY_PATH = "/v1/graphql"

client = weaviate.Client(
    url="https://",
)

async def init_db():
    try:
        async with aiosqlite.connect(DB_NAME) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trideque_point INT,
                    response TEXT,
                    response_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INT
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trideque_point INT,
                    summarization_context TEXT,
                    full_text TEXT
                )
            """)

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
        def find_overlap(chunk, next_chunk):
            max_overlap = min(len(chunk), 300)
            for overlap in range(max_overlap, 0, -1):
                if chunk.endswith(next_chunk[:overlap]):
                    return overlap
            return 0

        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        prompt_chunks = [prompt[i:i + chunk_size] for i in range(0, len(prompt), chunk_size)]
        responses = []
        last_output = ""

        for i, chunk in enumerate(prompt_chunks):
            output_dict = llm(chunk, max_tokens=min(max_tokens, chunk_size))

            # Check if the output is a dictionary
            if not isinstance(output_dict, dict):
                logger.error(f"Output from Llama for chunk {i} is not a dictionary: {type(output_dict)}")
                continue

            choices = output_dict.get('choices', [])
            if not choices or not isinstance(choices[0], dict):
                logger.error(f"No valid choices in Llama output for chunk {i}")
                continue

            output = choices[0].get('text', '')
            if not output:
                logger.error(f"No text found in Llama output for chunk {i}")
                continue

            if i > 0 and last_output:
                overlap = find_overlap(last_output, output)
                output = output[overlap:]

            responses.append(output)
            last_output = output

            print(f"Processed output for chunk {i}: {output}")

        final_response = ''.join(responses)
        return final_response
    except Exception as e:
        logger.error(f"Error in llama_generate: {e}")
        return None



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
        self.executor = ThreadPoolExecutor(max_workers=4)
        
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

            if 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
                interactions = response['data']['Get']['InteractionHistory']

                processed_interactions = []
                for interaction in interactions:
                    user_message = interaction['user_message']
                    ai_response = interaction['ai_response']
                    summarized_interaction = summarizer.summarize(f"{user_message} {ai_response}")
                    sentiment = TextBlob(summarized_interaction).sentiment.polarity

                    processed_interactions.append({
                        "user_message": user_message,
                        "ai_response": ai_response,
                        "summarized_interaction": summarized_interaction,
                        "sentiment": sentiment
                    })

                result_queue.put(processed_interactions)
            else:
                logger.error("No interactions found for the given theme.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])




    def process_response_and_store_in_weaviate(self, user_message, ai_response):
        response_blob = TextBlob(ai_response)
        keywords = response_blob.noun_phrases
        sentiment = response_blob.sentiment.polarity

        interaction_object = {
            "userMessage": user_message,
            "aiResponse": ai_response,
            "keywords": list(keywords),
            "sentiment": sentiment
        }

        interaction_uuid = str(uuid.uuid4())

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

            summarized_context = summarizer.summarize(context)
        except Exception as e:
            print(f"Error in summarizing context: {e}")
            summarized_context = context

        try:

            sentiment = TextBlob(summarized_context).sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            sentiment = TextBlob("").sentiment

        positive_class_mappings = {
            "keyword1": "PositiveClassA",
                "keyword2": "PositiveClassB",

        }

        negative_class_mappings = {
            "keyword1": "NegativeClassA",
            "keyword2": "NegativeClassB",

        }


        default_mapping = {
            "keyword1": "NeutralClassA",
            "keyword2": "NeutralClassB",

        }


        if sentiment.polarity > 0:
            mapping = positive_class_mappings
        elif sentiment.polarity < 0:
            mapping = negative_class_mappings
        else:
            mapping = default_mapping
            
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
        coro = coro_func(message, result_queue)
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
            if response:
                response_text = response
                self.response_queue.put({'type': 'text', 'data': response_text})

                keywords = self.extract_keywords(message)
                mapped_classes = self.map_keywords_to_weaviate_classes(keywords, message)  # Assuming the message itself is the context

                self.create_interaction_history_object(message, response_text)
            else:
                logger.error("No response generated by llama_generate")

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")


    def on_submit(self, event=None):
        message = self.entry.get().strip()
        if message:
            # Insert the message into the chat box
            self.text_box.insert(tk.END, f"You: {message}\n")
            
            # Clear the input box
            self.entry.delete(0, tk.END)

            # Ensure the latest message is visible
            self.text_box.see(tk.END)

            # Continue with other functionalities like generating response
            self.executor.submit(self.generate_response, message)
            self.executor.submit(self.generate_images, message)
            self.after(100, self.process_queue)

    def create_object(self, class_name, object_data):

        unique_string = f"{object_data['time']}-{object_data['user_message']}-{object_data['ai_response']}"


        object_uuid = uuid.uuid5(uuid.NAMESPACE_URL, unique_string).hex


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
                    self.image_label.image = response['data']
                self.text_box.see(tk.END)
        except queue.Empty:
            self.after(100, self.process_queue)
            
    def extract_keywords(self, message):
        blob = TextBlob(message)
        nouns = blob.noun_phrases
        return list(nouns)

    async def retrieve_past_interactions(self, theme, result_queue):
        try:

            def sync_query():
                return self.client.query.get("interaction_history", ["user_message", "ai_response"]).with_near_text({
                    "concepts": [theme],
                    "certainty": 0.7
                }).do()


            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(executor, sync_query)

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
                        self.image_label.image = img_tk
                except ValueError as e:
                    print("Error processing image data: ", e)
            else:
                print("Error generating image: ", response.status_code)

        except Exception as e:
             logger.error(f"Error in generate_images: {e}")

    def setup_gui(self):
        self.title("OneLoveIPFS AI")
        self.geometry(f"{1100}x{580}")
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        logo_path = os.path.join(os.getcwd(), "logo.png")
        logo_img = Image.open(logo_path).resize((140, 77))  # Add the .resize() method with the desired dimensions
        logo_photo = ImageTk.PhotoImage(logo_img)  # Convert PIL.Image to tkinter.PhotoImage
        self.logo_label = tk.Label(self.sidebar_frame, image=logo_photo, bg=self.sidebar_frame["bg"])  # Create a tkinter.Label
        self.logo_label.image = logo_photo  # Keep a reference to the image
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))  # This is the correct position for the logo_label grid statement
        self.text_box = customtkinter.CTkTextbox(self, bg_color="white", text_color="white", border_width=0, height=20, width=50, font=customtkinter.CTkFont(size=13))
        self.text_box.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.entry = customtkinter.CTkEntry(self, placeholder_text="Chat With Llama")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.on_submit)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")
        self.entry.bind('<Return>', self.on_submit)
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
