import os

# ******************Remove these keys for sharing outside**************
os.environ[''] = ""

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == "__main__":
    print("Loding pdf file..")
    pdf_path = './2210.03629.pdf'

    # This loads and pdf file and chunks it
    # Load, read and chunk the doc
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, 
                                          chunk_overlap=30,
                                          separator="\n")
    
    docs = text_splitter.split_documents(documents=documents)
    
    embeddings = OpenAIEmbeddings()

    vectorestore = FAISS.from_documents(docs, embeddings)
    vectorestore.save_local("fiass_index_react")

     
    # Now for trying it out, we are going to load from the local storage device index react
    # That we saved in persisted and use it as vector store.

    new_vectorstore = FAISS.load_local("fiass_index_react", embeddings)

    # Now are have to write the chian that is going to sticth everything up together like
    # we saw in the earlier video. 

    # What below does is, take the query that we are going to send it + 
    # turn it into vector + send it into device vector store + Then its going to find
    # similar vectors to that vector + its going to bring those vectors back. We are going 
    # to translate them into the text and we are going send this as the context exactly like
    # VDBQA chain - but with different parameters.

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=new_vectorstore.as_retriever() # converting vector store into a retriever object
    )

    res = qa.run("Give me the gist of ReAct in 3 sentences")
    print(res)
