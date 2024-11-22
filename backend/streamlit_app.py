import os
import logging
from dotenv import load_dotenv
import streamlit as st
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from typing import Annotated

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables with validation
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

if not GROQ_API_KEY or not TAVILY_API_KEY:
    logging.error("Missing required API keys. Please check your .env file.")
    raise ValueError("Missing required API keys")

class State(TypedDict):
    """Defines the conversation state."""
    messages: Annotated[list, add_messages]

def setup_graph():
    """Set up the LangGraph state graph with error handling."""
    try:
        logging.info("Setting up the state graph...")
        graph_builder = StateGraph(State)

        # Initialize Tavily tool with improved configuration
        tool = TavilySearchResults(
            api_key=TAVILY_API_KEY,
            max_results=3,
            search_depth="advanced"
        )
        tools = [tool]
        logging.info("Tavily tool initialized successfully")

        # Configure LLM without tool binding
        llm = ChatGroq(
            model="gemma-7b-it",
            temperature=0.7,  # Increased temperature for more natural responses
            api_key=GROQ_API_KEY
        )
        logging.info("ChatGroq LLM initialized successfully")

        def chatbot(state: State):
            """Enhanced chatbot function."""
            try:
                logging.debug(f"Processing state with messages: {state['messages']}")
                response = llm.invoke(state["messages"])
                logging.debug(f"LLM response received: {response}")
                return {"messages": [response]}
            except Exception as e:
                logging.error(f"Error in chatbot function: {str(e)}", exc_info=True)
                raise

        # Build graph with simplified flow (removed tool integration for now)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")

        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        logging.info("State graph setup completed successfully")
        return graph

    except Exception as e:
        logging.error(f"Error setting up graph: {str(e)}", exc_info=True)
        raise

def get_prompt(user_input):
    """Generate prompt with improved system message."""
    if not user_input or not user_input.strip():
        logging.warning("Empty user input received")
        return None
    
    logging.debug(f"Generating prompt for input: {user_input}")
    return [
        {
            'role': 'system',
            'content': '''You are an AI car research assistant specializing in the Indian car market. 
            Provide detailed information about cars, including specifications, features, performance, 
            and market analysis. Focus on being informative and accurate.'''
        },
        {
            'role': 'user',
            'content': user_input
        }
    ]

# Streamlit UI
st.set_page_config(page_title="CarBot Chat", layout="centered")
st.title("CarBot Chat")
st.write("Ask me anything about cars in the Indian market!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    try:
        st.session_state.graph = setup_graph()
        logging.info("Session state initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize session state: {str(e)}", exc_info=True)
        st.error("Failed to initialize chat application. Please check the logs.")
        st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**CarBot:** {message['content']}")

# Input handling with proper state management
user_input = st.text_input(
    "Type your message here:",
    key="user_input_field",  # Changed key to avoid conflicts
    max_chars=500
)

# Handle send button click
if st.button("Send", key="send_button"):
    if user_input.strip():
        try:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate and validate prompt
            prompt = get_prompt(user_input)
            if not prompt:
                raise ValueError("Invalid prompt generated")

            # Process through graph
            logging.info("Processing message through graph...")
            config = {"configurable": {"thread_id": "1"}}
            
            events = st.session_state.graph.stream(
                {"messages": [
                    ("system", prompt[0]['content']),
                    ("user", prompt[1]['content'])
                ]},
                config,
                stream_mode="values"
            )

            # Process response
            ai_response = None
            for event in events:
                logging.debug(f"Event received: {event}")
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    ai_response = last_message.content

            if not ai_response:
                raise ValueError("No response generated by the AI model")

            # Add response to history
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            logging.info("Message processed successfully")

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logging.error(error_msg, exc_info=True)
            st.error("Sorry, there was an error processing your message. Please try again.")
        
        # Use rerun instead of directly modifying session state
        st.rerun()