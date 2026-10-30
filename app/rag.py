from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    DISTANCE_THRESHOLD,
)


def load_vectorstore():
    """Load the existing Chroma vector database."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    return vectorstore


def retrieve_documents(question):
    """Retrieve relevant document chunks using a relevance threshold
    and remove exact duplicate chunks.
    """

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    relevant_documents = []

    # Keep track of chunks we have already added.
    seen_chunks = set()

    for document, score in results:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        print(
            f"Retrieved: "
            f"{source} "
            f"Page {page} "
            f"Distance: {score:.3f}"
        )

        #**********************
        # Distance filtering  *
        #**********************

        if score > DISTANCE_THRESHOLD:
            continue

        #************************
        # Duplicate filtering   *
        #************************

        chunk_text = document.page_content.strip()

        if chunk_text in seen_chunks:
            print(
                f"Skipped duplicate chunk: "
                f"{source} Page {page}"
            )
            continue

        seen_chunks.add(chunk_text)

        relevant_documents.append(document)

    return relevant_documents

def build_context(documents):
    """Build context for the LLM with source identifiers."""

    context_parts = []

    for i, document in enumerate(documents, start=1):
        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {i}
Paper: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, context):
    """Generate a grounded answer using Qwen."""

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI research assistant specializing in
computer vision and machine learning research papers.

Your task is to answer the user's question using ONLY
the research paper context provided below.

IMPORTANT RULES:

1. Use ONLY information explicitly supported by the
   provided context.

2. Do NOT use general knowledge, prior knowledge, or
   information from outside the provided context.

3. Do NOT invent facts, explanations, equations,
   numbers, definitions, examples, comparisons, or
   conclusions.

4. Before answering, determine whether the provided
   context actually contains information that directly
   answers the user's question.

5. Retrieved context may contain documents that are
   related to the general topic but do NOT directly
   answer the question. Do not treat the presence of a
   related document as sufficient evidence.

6. If the context does not contain enough information
   to answer the question, respond exactly with:

   "The available research papers do not contain enough
   information to answer this question."

7. If the context only partially answers the question,
   do NOT fill the missing information using general
   knowledge. Instead, use the refusal sentence from
   Rule 6.

8. When the context contains enough information, provide
   a concise and technically accurate answer.

9. Preserve the important technical terminology used
   in the research papers.

10. When the paper explicitly uses a technical term,
    prefer that exact terminology instead of replacing
    it with a general synonym.

    For example:
    - Use "self-attention" when the context uses
      "self-attention".
    - Use "convolutional layers" when the context uses
      "convolutional layers".
    - Use "Transformer" when the context uses
      "Transformer".
    - Use "residual function" when the context uses
      "residual function".

11. Preserve mathematical definitions exactly as they
    appear in the context.

    For example, if the context defines:

        F(x) := H(x) - x

    do not change the meaning of F(x), H(x), or x.

12. Do not interpret a mathematical function differently
    from the definition given in the context.

13. Do not claim that a function, layer, architecture,
    mechanism, or component performs a purpose unless
    that purpose is supported by the context.

14. When the question asks "why", explain the reason only
    if that reason is supported by the provided context.

15. When the question asks "how", describe only the steps
    or mechanism that are supported by the context.

16. When the question asks for a comparison, explicitly
    identify the differences requested in the question.

17. For comparison questions, organize the answer around
    the relevant characteristics supported by the context.

18. Do not introduce unrelated information simply because
    it appears in the retrieved documents.

19. Do not make claims about performance, datasets,
    computational efficiency, architecture, training,
    memory usage, or other characteristics unless they
    are supported by the retrieved context.

20. If multiple papers are relevant, combine information
    from them only when the context directly supports the
    connection.

21. Do NOT create a Sources section.

22. Do NOT mention papers or documents that were not useful
    for answering the question.

23. Keep the answer concise. Normally provide one or a few
    focused paragraphs or a short numbered list when
    appropriate.

24. Do not repeat the same point in different wording.

25. The retrieved context is the ONLY source of truth for
    this answer.

Research paper context:

{context}

User question:

{question}

Answer:
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return response.content.strip()


def ask_question(question):

    print("\nRetrieving relevant documents...")

    documents = retrieve_documents(question)

    if not documents:

        return (
            "The available research papers do not contain "
            "enough information to answer this question.",
            []
        )

    context = build_context(documents)

    answer = generate_answer(
        question,
        context,
    )

    return answer, documents


if __name__ == "__main__":

    question = input(
        "\nAsk a question about the research papers: "
    )

    answer, documents = ask_question(question)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for i, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page",
            "Unknown",
        )

        print(
            f"{i}. {source} — Page {page}"
        )