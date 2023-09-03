import os
import streamlit as st
import sys
from utils.helper import load_document,chunk_data,create_embeddings
from llms import clear_history,ask_and_get,calculate_embedding_cost

if __name__=="__main__":
    st.image("img/img.png")
    # st.image("img/LangChain_&_HuggingFace.png")
    st.subheader('LLM Question-Answering Application 🤖')

    with st.sidebar:
        
        api_key = st.text_input('OpenAI API Key:', type='password')
        # if api_key:
        #     os.environ['OPENAI_API_KEY'] = api_key
        if ["openai_api_key"] not in st.session_state:
            st.session_state.openai_api_key = api_key
        # file upload widget
        uploaded_file = st.file_uploader("Upload a file :", type=[".pdf", ".docx",".txt"])

        # Chunk size value widget
        chunk_size = st.number_input("Chunk_size :",min_value=150, max_value=1000,value=512,on_change=clear_history)

        # temperature value for LLM model widget
        temperature = st.number_input("temperature :",min_value=0.0, max_value=1.0,value=0.1,on_change=clear_history)

        # Max token value widget for LLM
        # max_length = st.number_input("max_token_length :",min_value=256, max_value=1028,value=512,on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner("Reading.. Chunking... Embedding.... file."):
                bytes_data = uploaded_file.read()
                os.makedirs("./files",exist_ok=True)
                filename = os.path.join("./files",uploaded_file.name)
                
                with open(filename,"wb") as f:
                    f.write(bytes_data)
                
                data = load_document(filename)
                chunks = chunk_data(data,chunk_size=chunk_size)
                st.write(f" Chunks size {chunk_size} Chunks len {len(chunks)}")


                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks,st.session_state.openai_api_key)

                st.session_state.vs = vector_store

                st.success("file uploaded, chunked and embedded successfully.")
        
    q = st.text_input("Ask a question about the file you uploaded.")

    # If question is asked:
    if q:
        if st.session_state.vs: # if vectorstore in the session (if file uploaded )
            vector_store = st.session_state.vs
            st.write(f"k : {k}")

            # llm = load_llm()

            # Question to the LLM with Vector store and retrieval doc value of "k".
            answer = ask_and_get(vector_store,q,st.session_state.openai_api_key,k)

            st.text_area("LLM answer :", value=answer)

            st.divider()

            # Check chat history if not there create one.
            if "history" not in st.session_state:
                st.session_state.history = ""
            
            # Current ques and Ans.
            value = f"Q : {q} \nA: {answer}"


            st.session_state.history = f'{ value } \n { "-" * 100 } \n { st.session_state.history }'
            h = st.session_state.history

            # History for display
            st.text_area(label="Chat history : ",value=h,key="history",height=400)


















