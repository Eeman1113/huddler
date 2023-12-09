import streamlit as st
from youtuber import fetch_youtube_captions
from agent import process_and_cluster_captions, generate_summary, answer_question, reset_globals

# Set Streamlit page configuration with custom tab title
st.set_page_config(page_title="Huddler", page_icon="🏄", layout="wide")

def user_query(question, openai_api_key, model_name):
    """Process and display the query response."""
    # Add the user's question to the conversation
    st.session_state.conversation.append((f"{question}", "user-message"))

    # Check if this query has been processed before
    if question not in st.session_state.processed_questions:
        # Process the query
        answer = answer_question(question, openai_api_key, model_name)
        if isinstance(answer, str):
            st.session_state.conversation.append((f"{answer}", "grimoire-message"))
        else:
            st.session_state.conversation.append(("Could not find a proper answer.", "grimoire-message"))
        
        st.rerun()

        # Mark this question as processed
        st.session_state.processed_questions.add(question)


# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
    st.session_state.asked_questions = set()
    st.session_state.processed_questions = set()

# Sidebar for input and operations
with st.sidebar:
    st.title("Meet Me Hun Mai")
    # st.image("img.png") 

    # Expandable Instructions
    with st.expander("🔍 How to use:", expanded=False):
        st.markdown("""
            - 🔐 **Enter your OpenAI API Key.**
            - 📺 **Paste a YouTube URL.**
            - 🏃‍♂️ **Click 'Run it' to process.**
            - 🕵️‍♂️ **Ask questions in the chat.**
        """)

    # Model selection in the sidebar
    model_choice = st.sidebar.selectbox("Choose Model:", 
                                        ("GPT-4 Turbo", "GPT-3.5 Turbo"), 
                                        index=0)  # Default to GPT-4 Turbo

    # Map friendly names to actual model names
    model_name_mapping = {
        "GPT-4 Turbo": "gpt-4-1106-preview",
        "GPT-3.5 Turbo": "gpt-3.5-turbo"
    }

    selected_model = model_name_mapping[model_choice]
    st.session_state['selected_model'] = model_name_mapping[model_choice]


    # Input for OpenAI API Key
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    # Save the API key in session state if it's entered
    if openai_api_key:
        st.session_state['openai_api_key'] = openai_api_key

    youtube_url = st.text_input("Enter YouTube URL:")

    # Button to trigger processing
    if st.button("🚀Run it"):
        if openai_api_key:
            if youtube_url and 'processed_data' not in st.session_state:
                reset_globals()
                with st.spinner('👩‍🍳 GPT is cooking up your meet... hang tight for a few secs🍳'):
                    captions = fetch_youtube_captions(youtube_url)
                    if captions:
                        representative_docs = process_and_cluster_captions(captions, st.session_state['openai_api_key'])
                        summary = generate_summary(representative_docs, st.session_state['openai_api_key'], selected_model)
                        st.session_state.processed_data = (representative_docs, summary)
                        if 'summary_displayed' not in st.session_state:
                            st.session_state.conversation.append((f"Here's a rundown of the conversation: {summary}", "summary-message"))
                            guiding_message = "Feel free to ask me anything else about it! :)"
                            st.session_state.conversation.append((guiding_message, "grimoire-message"))
                            st.session_state['summary_displayed'] = True
                    else:
                        st.error("Failed to fetch captions.")
        else:
            st.warning("Please add the OpenAI API key first.")


# Main app logic
for message, css_class in st.session_state.conversation:
    role = "assistant" if css_class in ["grimoire-message", "summary-message", "suggestion-message"] else "user"
    with st.chat_message(role):
        st.markdown(message)


# Chat input field
if prompt := st.chat_input("Ask me anything about the podcast..."):
    user_query(prompt, st.session_state.get('openai_api_key', ''), st.session_state.get('selected_model', 'gpt-4-1106-preview'))
