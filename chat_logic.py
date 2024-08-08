import os
import google.generativeai as genai
from langgraph.graph import StateGraph
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# Ensure the API key is set in the environment variables
if 'API_KEY' not in os.environ:
    raise KeyError("API_KEY environment variable is not set")

# Configure the Gemini API
genai.configure(api_key=os.environ['API_KEY'])

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

# Define the state structure
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the StateGraph
graph_builder = StateGraph(State)

# Function to check if the message is related to video games
def is_video_game_related(message):
    video_game_keywords = ["game", "play", "console", "PC", "Xbox", "PlayStation", "Nintendo", "Steam", "eSports"]
    return any(keyword in message.lower() for keyword in video_game_keywords)

# Define the responses
def get_video_game_response(state):
    message_content = state["messages"][-1].content
    if is_video_game_related(message_content):
        try:
            # Use the Gemini API to generate a response
            response = model.generate_content(message_content)
            return {"messages": [{"role": "assistant", "content": response.text}]}
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return {"messages": [{"role": "assistant", "content": "I'm having trouble connecting to the game server. Please try again later."}]}
    else:
        return {"messages": [{"role": "assistant", "content": "I only talk about video games."}]}

# Add nodes to the graph
graph_builder.add_node("chatbot", get_video_game_response)

# Set entry and finish points
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph
graph = graph_builder.compile()

# Function to interact with the chatbot
def chat_with_bot(user_message):
    for event in graph.stream({"messages": [{"role": "user", "content": user_message}]}):
        for value in event.values():
            return value["messages"][-1]["content"]

# Example usage
if __name__ == "__main__":
    print("Chatbot is running. Type your message:")
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        response = chat_with_bot(user_message)
        print(f"Bot: {response}")
