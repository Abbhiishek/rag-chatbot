import os
from io import BytesIO
from typing import Tuple, List

from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

def parse_mdx(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    content = file.read().decode('utf-8')
    # You might want to add more sophisticated MDX parsing here
    # print("Parsing MDX file:", filename)
    return [content], filename

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    doc_chunks = []
    for i, page in enumerate(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        chunks = text_splitter.split_text(page)
        for j, chunk in enumerate(chunks):
            # Calculate the start and end line numbers for this chunk
            start_line = page[:page.index(chunk)].count('\n') + 1
            end_line = start_line + chunk.count('\n')
            
            doc = Document(
                page_content=chunk, 
                metadata={
                    "chunk": j,
                    "source": f"{filename}:{start_line}-{end_line}",
                    "filename": filename,
                    "start_line": start_line,
                    "end_line": end_line
                }
            )
            doc_chunks.append(doc)
    return doc_chunks


def get_index_for_mdx(mdx_files, mdx_names, openai_api_key):
    documents = []
    for mdx_file, mdx_name in zip(mdx_files, mdx_names):
        text, filename = parse_mdx(BytesIO(mdx_file), mdx_name)
        # print("Text to docs:", filename)
        documents = documents + text_to_docs(text, filename)

    # print("Embedding documents:", documents)
    
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint="https://chatsupportsys5416848984.cognitiveservices.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
        openai_api_key=openai_api_key,
        model ="text-embedding-3-large",
        chunk_size=1,  # You can adjust this value as needed
    )
    # print("FAISS index:", embeddings)
    
    index = FAISS.from_documents(documents, embeddings)
    return index