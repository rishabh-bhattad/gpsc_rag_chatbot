import streamlit as st
from chatbot import query_rag

# Page Configuration
st.set_page_config(page_title="GPSC AI Secretary", page_icon="🤖")
st.title("GPSC Meeting Assistant")

# Initialize Chat History
# We use session_state to keep memory between re-runs
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the GPSC AI Secretary. Ask me about funding, policies, or meeting minutes."}
    ]

# Display Chat Messages
# This loop re-draws the history every time the app updates
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question about GPSC records..."):
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate & Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching minutes..."):
            # Call backend
            response = query_rag(user_question=prompt)
            
            # Display the answer
            st.markdown(response['answer'])
            
            # Display Sources in a collapsible box
            if response['context']:
                with st.expander("View Sources"):
                    # De-duplicate sources for cleaner display
                    seen_sources = set()
                    for item in response['context']:
                        source_key = item['source']
                        if source_key not in seen_sources:
                            st.markdown(f"- **{item['source']}** ({item['date']})")
                            seen_sources.add(source_key)
            else:
                st.caption("No specific documents cited.")

    # Add Assistant Response to history (so it stays on screen)
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})