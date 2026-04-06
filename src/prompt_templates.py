"""RAG prompt templates for question answering and query rephrasing."""

RAG_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
1. Answer the question using ONLY the information in the context below.
2. If the context does not contain enough information to answer, say: "I don't have enough information in the uploaded documents to answer this question."
3. DO NOT make up or infer information that is not explicitly stated in the context.
4. Always cite your sources by mentioning the document name and page number.
5. Be concise and direct in your answers.
6. If multiple documents contain relevant information, synthesize them and cite all sources.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

ANSWER:"""


REPHRASE_PROMPT = """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that includes all necessary context.

If the question is already standalone and doesn't reference the conversation, return it unchanged.

CONVERSATION HISTORY:
{chat_history}

FOLLOW-UP QUESTION: {question}

STANDALONE QUESTION:"""


CONTEXT_TEMPLATE = """[Source: {source}, Page: {page}]
{text}
---"""


def build_context(search_results: list[dict]) -> str:
    """Format search results into a context string for the RAG prompt."""
    if not search_results:
        return "No relevant context found."

    context_parts = []
    for result in search_results:
        context_parts.append(
            CONTEXT_TEMPLATE.format(
                source=result["source"],
                page=result["page"],
                text=result["text"],
            )
        )
    return "\n".join(context_parts)


def build_rag_prompt(
    question: str,
    search_results: list[dict],
    chat_history: str = "",
) -> str:
    """Build the full RAG prompt with context and question."""
    context = build_context(search_results)
    return RAG_PROMPT.format(
        context=context,
        chat_history=chat_history or "No previous conversation.",
        question=question,
    )


def build_rephrase_prompt(question: str, chat_history: str) -> str:
    """Build prompt for rephrasing a follow-up question into a standalone one."""
    return REPHRASE_PROMPT.format(
        chat_history=chat_history,
        question=question,
    )
