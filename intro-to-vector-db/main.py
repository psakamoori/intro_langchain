import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain import VectorDBQA, OpenAI
import pinecone


pinecone.init(api_key="3519a6a1-d88b-449b-8699-1b03e76f1eed", environment="gcp-starter")

# ******************Remove these keys for sharing outside**************
os.environ['OPENAI_API_KEY'] = "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"

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("./mediumblogs/mediumblog1.txt")

    # Coverting above text file into a document
    document = loader.load()
    print(document)

    # Above document is big, we want to split it
    # Note: chunk_size and chunk_overlap are need to tuned according to the needs
    # If the prompt is not is responding to the way is should, it could be as a 
    # result of worng chunking size or wrong chunk overlap
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(document) # we are calling it document (even though its a list with one document)
    print(len(texts))

    # Now we will look into embedding part (on above chunks of text can be represented as vectors)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Now we want to take the texts, we want to embed them and we want to send them itno Pinecone.
    docsearch = Pinecone.from_documents(texts, 
                                        embeddings, 
                                        index_name="medium-blogs-data-index") 

    # Now lets import chains 
    # What RetreivalQA chain does is, Take the query (the prompt) it embeds it as a vector, 
    # then takes this vector, and by using the vector database that we are  using here is Pinecone
    # It take the query vector and then it plots it into the vector db and vector db then returns
    # us a couple of vectors which are closest to the query vector sematically, and the vector db will
    # return those neighboring vectors. 

    # Now chain is going to take those vectors, translate them back into chunks, and those chunks are
    # the relevant chunks and those are the context which we are going to pass to the modified prompt.
    # New prompt = Original prompt + the relevant chunks 
    # Basically, it will tell the LLM, hey we have this prompt - now for this prompt we know that those
    # texts are very, very important and they probably have the naswer for this prompt.

    # So please do your magic and use them as your context and the process where we take those embeddings 
    # and simply plug it in without doing anything into the prompt. This method is called stuffing.
    # Its also a method for, saving tokens (see this in detail in section 7). 

    # chain_type "stuff" this means that the context we are going to get from the vectro store, we simply
    # going to sutff it in the prompt (we are not going to do any transformations).

    # For 2nd argument, why are not we passing the vector store directly to this chian? and it has to go t
    # through the as retriever object first, looks like a redundant work. Answer for this is that retrieving 
    # relevant context to the LLM is something that can happen in couple of ways. We are seeing right now a 
    # retrieval from a vector store that uses semantic search but there are other types of retreivals and LC
    # abstracs, all of them for us.
    #qa = RetrievalQA.from_chain_type(
    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", 
        vectorstore=docsearch,
        return_source_documents=True
    )

    query ="What is a vector DB? Give me a 15 word answer for a begginer"
    result = qa({"query": query})
    print(result)

