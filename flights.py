import vertexai
import streamlit as st
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, Content, ChatSession
from services.flight_manager import search_flights, book_flight  # Import both functions

# Initialize Vertex AI with your project
project = "gemini-flights-433405"
vertexai.init(project=project)

# Define the flight search function using FunctionDeclaration
get_search_flights = generative_models.FunctionDeclaration(
    name="get_search_flights",
    description="Tool for searching a flight with origin, destination, and departure date",
    parameters={
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "The airport of departure for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "destination": {
                "type": "string",
                "description": "The airport of destination for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "departure_date": {
                "type": "string",
                "format": "date",
                "description": "The date of departure for the flight in YYYY-MM-DD format"
            },
        },
        "required": [
            "origin",
            "destination",
            "departure_date"
        ]
    },
)

# Define the book flight function using FunctionDeclaration
book_flight_declaration = generative_models.FunctionDeclaration(
    name="book_flight",
    description="Tool for booking a flight with origin, destination, departure date, passenger name, and payment details",
    parameters={
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "The airport of departure for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "destination": {
                "type": "string",
                "description": "The airport of destination for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "departure_date": {
                "type": "string",
                "format": "date",
                "description": "The date of departure for the flight in YYYY-MM-DD format"
            },
            "passenger_name": {
                "type": "string",
                "description": "Name of the passenger booking the flight"
            },
            "payment_details": {
                "type": "object",
                "description": "Payment details including card number, expiry date, and CVV",
                "properties": {
                    "card_number": {
                        "type": "string",
                        "description": "Credit card number"
                    },
                    "expiry_date": {
                        "type": "string",
                        "description": "Expiry date of the credit card in MM/YY format"
                    },
                    "cvv": {
                        "type": "string",
                        "description": "CVV code of the credit card"
                    }
                },
                "required": [
                    "card_number",
                    "expiry_date",
                    "cvv"
                ]
            }
        },
        "required": [
            "origin",
            "destination",
            "departure_date",
            "passenger_name",
            "payment_details"
        ]
    },
)

# Instantiate a Tool class encapsulating the function declarations
search_tool = generative_models.Tool(
    function_declarations=[get_search_flights, book_flight_declaration],
)

# Create a GenerationConfig object to set the generation parameters
config = generative_models.GenerationConfig(temperature=0.4)

# Initialize the GenerativeModel with the specified configuration and tools
model = GenerativeModel(
    "gemini-pro",
    tools=[search_tool],
    generation_config=config
)

# Step 1: Helper function to unpack and handle responses
def handle_response(response):
    # Check if the response contains a function call with arguments
    if response.candidates[0].content.parts[0].function_call.args:
        # Extract the function call arguments
        response_args = response.candidates[0].content.parts[0].function_call.args
        
        function_params = {}
        for key in response_args:
            value = response_args[key]
            function_params[key] = value
        
        # Execute the appropriate function based on the function call name
        if response.candidates[0].content.parts[0].function_call.name == "get_search_flights":
            results = search_flights(**function_params)
        elif response.candidates[0].content.parts[0].function_call.name == "book_flight":
            results = book_flight(**function_params)
        else:
            results = None
        
        if results:
            # Send the results back to the chat session
            intermediate_response = chat.send_message(
                Part.from_function_response(
                    name=response.candidates[0].content.parts[0].function_call.name,
                    response=results
                )
            )
            # Return the text part of the response
            return intermediate_response.candidates[0].content.parts[0].text
        else:
            return "Operation Failed"
    else:
        # If no function call, just return the text part of the response
        return response.candidates[0].content.parts[0].text

# Step 2: Helper function to display and send Streamlit messages
def llm_function(chat: ChatSession, query):
    response = chat.send_message(query)
    output = handle_response(response)
    
    with st.chat_message("model"):
        st.markdown(output)
    
    # Store the conversation in session state
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    st.session_state.messages.append(
        {
            "role": "model",
            "content": output
        }
    )

# Step 3: Streamlit UI setup
st.title("Gemini Flights")

# Start a chat session with the model
chat = model.start_chat()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for index, message in enumerate(st.session_state.messages):
    content = Content(
        role=message["role"],
        parts=[Part.from_text(message["content"])]
    )
    
    if index != 0:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat.history.append(content)

# Initial message startup
if len(st.session_state.messages) == 0:
    # Invoke an initial message to introduce the assistant
    initial_prompt = "Introduce yourself as a flights management assistant, ReX, powered by Google Gemini and designed to search/book flights. You use emojis to be interactive. For reference, the year for dates is 2024"
    llm_function(chat, initial_prompt)

# Capture user input
query = st.chat_input("Gemini Flights")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    llm_function(chat, query)
