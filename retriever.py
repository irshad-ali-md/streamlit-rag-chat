from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import CSVLoader

from pinecone import Pinecone
import os
import warnings
warnings.filterwarnings('ignore')

def retrieve_from_pinecone(user_query="Who was president of Notre Dame?"):
    ## Pinecone context code:
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    index_name = "test"

    # connect to index
    index = pc.Index(index_name)

    # view index stats
    index.describe_index_stats()

    ## Use this to upload documents as vectors

    # loader = CSVLoader("./ingestion/squad-train.csv", encoding="utf8")

    # documents = loader.load()
    # pinecone = PineconeVectorStore.from_documents(
    #     documents[:100], embeddings, index_name=index_name
    # )

    ### Use this to retrieve from existing vector store
    pinecone = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

    context= pinecone.similarity_search(user_query)[:3]
    return context

#print(retrieve_from_pinecone())