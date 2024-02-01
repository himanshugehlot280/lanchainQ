from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import getpass

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 📈")

   
    
    pdf_file = st.file_uploader("Upload a pdf file", type="pdf")
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    #print(pages);
    if pdf_file is not None:       

        faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
        docs = faiss_index.similarity_search("How will the community be engaged?", k=2)
    for doc in docs:
       print(str(doc.metadata["page"]) + ":", doc.page_content[:300])



        #agent = create_csv_agent(
        #    OpenAI(temperature=0), pdf_file, verbose=True)

    user_question = st.text_input("Ask a question about your PDF: ")

    if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(faiss_index.run(user_question))


if __name__ == "__main__":
    main()
