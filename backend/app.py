from flask import Flask, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from typing import Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
logging.info("Environment variables loaded.")

# Initialize the Flask app
app = Flask(__name__)
logging.info("Flask app initialized.")

# Retrieve API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
logging.info("API keys loaded from environment.")

class State(TypedDict):
    """
    A typed dictionary defining the state for the AI assistant.

    Attributes:
        messages (list): A list of messages representing the conversation history with the assistant.
    """
    messages: Annotated[list, add_messages]

def setup_graph():
    """
    Set up and compile the state graph to process messages and manage conversation flow.

    Returns:
        StateGraph: A compiled graph managing state and invoking the AI model.
    """
    logging.info("Setting up the state graph...")
    graph_builder = StateGraph(State)

    # Initialize the Tavily tool
    tool = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
    tools = [tool]

    # Configure the LLM (ChatGroq) and bind it with tools
    llm = ChatGroq(model="gemma-7b-it", temperature=0, api_key=GROQ_API_KEY)
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        """
        Invoke the LLM with the current state.

        Args:
            state (State): The current conversation state.

        Returns:
            dict: The LLM's response messages.
        """
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Set up nodes and transitions
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()
    logging.info("State graph setup complete.")
    return graph_builder.compile(checkpointer=memory)

graph = setup_graph()

def get_prompt(user_input):
    """
    Construct a prompt for the AI assistant based on user input.

    Args:
        user_input (str): The input message from the user.

    Returns:
        list: A structured prompt for the assistant's response, including system and user roles.
    """
    logging.debug(f"Generating prompt for user input: {user_input}")
    prompt = [
        {
    'role': 'system',
    'content': '''
    You are an AI car research assistant specializing in the Indian car market.
    Your task is to provide detailed, factual, and objective responses to user queries in plain text.
    The responses should be direct, without any special formatting (such as bullet points or asterisks).
    Focus only on key details such as car specifications, performance, and features, in a clear, concise, and plain text format.
    Do not include introductions, conclusions, references, or any extra explanations.
    '''
        },
        {
    'role': 'user',
    'content': f'''
    Information: """{user_input}"""
    Provide a direct, detailed response to the following query: """{user_input}"""
    Focus on the essential details, such as specifications, performance, and features, and present them as plain text.
    Do not use any bullet points, asterisks, or other special characters.
    '''
        }
    ]
    logging.info("Prompt generated.")
    return prompt

@app.route('/')
def home():
    """
    Render the homepage of the Flask app.

    Returns:
        Response: The rendered HTML for the index page.
    """
    logging.info("Rendering homepage.")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests from the front end, invoking the LLM and returning a response.

    Receives user input via a POST request, processes it through the state graph, and sends back the AI's response.

    Returns:
        Response: JSON containing the AI's response.
    """
    user_input = request.json.get('message')
    logging.info(f"Received user input: {user_input}")

    # Construct the prompt based on user input
    prompt = get_prompt(user_input)
    system_message = prompt[0]['content']
    user_message = prompt[1]['content']

    try:
        # Process the input through the graph and return the AI's response
        config = {"configurable": {"thread_id": "1"}}
        events = graph.stream(
            {"messages": [("system", system_message), ("user", user_message)]}, config, stream_mode="values"
        )

        ai_response = None
        for event in events:
            last_message = event["messages"][-1]
            ai_response = last_message.content
            logging.debug(f"Event processed: {event}")

        # Check if AI response was obtained
        if not ai_response:
            raise ValueError("No response generated by the AI model.")

    except Exception as e:
        logging.error("Error while processing the chat request", exc_info=True)
        ai_response = "Sorry, there was an issue processing your request. Please try again later."

    logging.info(f"AI response generated: {ai_response}")
    return jsonify({'response': ai_response})

if __name__ == '__main__':
    """
    Entry point of the application. Starts the Flask server.
    """
    app.run(debug=True)
    logging.info("Flask app started.")
