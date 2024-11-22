# CarBot Project Documentation

## Overview

**CarBot** is an AI-powered chatbot designed to guide users in making informed car-buying decisions. Utilizing advanced natural language processing and large language models (LLMs), CarBot assists users by providing relevant, accurate, and detailed information about cars available in India, helping them navigate features, specifications, and comparisons with ease.

## Project Structure

The project consists of a **Flask web application** that integrates various tools and APIs to manage the chatbot's conversational flow and provide a user-friendly experience. The following libraries and tools are used:
- **Flask**: Serves as the web framework for handling user requests and routing.
- **LangGraph**: Manages the conversational state graph, enabling conditional logic based on user queries.
- **ChatGroq**: An LLM for generating responses, using pre-trained models suitable for car-related queries.
- **Tavily Search**: Integrated as a tool for gathering up-to-date car information.

## Features

- **Car Information Retrieval**: Users can query CarBot for detailed specifications, comparisons, and recommendations.
- **State-Driven Conversation**: The chatbot utilizes a state graph to maintain conversational flow and context.
- **Error Handling and Logging**: Comprehensive logging and error handling ensure smooth operations and simplify debugging.
- **User-Friendly Interface**: A simple chat interface provides an intuitive way for users to interact with CarBot.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd CarBot
   ```

2. **Set up the Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add the following keys with your API credentials:
     ```plaintext
     GROQ_API_KEY=your_groq_api_key
     TAVILY_API_KEY=your_tavily_api_key
     ```

5. **Run the Application**:
   ```bash
   flask run
   ```

## Usage

1. Open your browser and navigate to `http://127.0.0.1:5000`.
2. Enter your query in the chat interface. CarBot will generate a response based on its car-related knowledge base and provide details relevant to car buying in India.

## Code Overview

### 1. `app.py`
The main application file, containing:
   - Flask routes and configuration.
   - Prompt generation function.
   - Integration with **LangGraph** and **ChatGroq** for conversational state management.

### 2. `setup_graph()`
A function to initialize the **LangGraph** state graph, configuring the conversational logic and managing tool nodes for handling specific user intents.

### 3. `get_prompt(user_input)`
Generates a structured prompt for the LLM based on user queries. CarBot uses a system message to maintain its role as a car research assistant.

### 4. `/chat` Route
The primary endpoint for processing user inputs, passing the prompt through the **LangGraph** state graph, and returning the AI’s response.

## Logging and Error Handling

The project integrates logging and exception handling to enhance maintainability:
- Logs are written to `app.log` and the console, tracking key events and errors.
- Detailed error handling in the `/chat` route ensures that users receive feedback even if processing errors occur.

## Project Goals

The aim of CarBot is to become a reliable resource for prospective car buyers in India, offering structured, accessible, and comprehensive car-related insights to support informed decision-making.

## Future Enhancements



---