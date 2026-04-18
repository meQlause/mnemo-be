from langchain_core.prompts import PromptTemplate

# Prompt for when no context is found in the notes
no_context_prompt = PromptTemplate.from_template(
    """You are a helpful assistant with access to the user's personal notes.
The user has asked a question, but NO relevant information was found in their notes.

Instructions:
1. Politely inform the user that you have no context from their notes to answer this specific question.
2. Remind them that you are restricted to answering based ONLY on their personal notes.
3. DO NOT provide any information from your external general knowledge.
4. DO NOT answer the question using outside knowledge.

Conversation History:
{history}

Question:
{input}

Answer:"""
)

# Strict RAG prompt for the FIRST successful context hit
rag_initial_prompt = PromptTemplate.from_template(
    """You are a helpful assistant answering questions based ONLY on the user's personal notes.

Instructions:
1. Answer the question using ONLY the provided "Context from Personal Notes".
2. You MUST NOT use any outside information or general knowledge.
3. If the answer is not in the context, say that you don't have enough context in your notes.
4. Be concise and professional.
5. If you use bullet points, they MUST always start with an asterisk (* ).

Context from Personal Notes:
{context}

Conversation History:
{history}

Question:
{input}

Answer:"""
)

# Flexible RAG prompt for follow-up questions once context is established
rag_followup_prompt = PromptTemplate.from_template(
    """You are a helpful assistant discussing the user's personal notes. 
A context has already been established from their notes.

Instructions:
1. While your answers should be primarily grounded in the provided "Context from Personal Notes" and the previous conversation history, you are free to discuss broader implications and use your general knowledge to provide a richer, more helpful response.
2. Ensure you remain consistent with the information found in the notes.
3. If the user asks something completely unrelated to the notes or current context, steer them back or politely mention the notes-only restriction if appropriate.

Context from Personal Notes:
{context}

Conversation History:
{history}

Question:
{input}

Answer:"""
)

title_prompt = PromptTemplate.from_template(
    """Given the following text, generate a short, concise title (max 5 words).
Do not include quotes or any other formatting.

Text:
{text}

Title:"""
)

analyze_prompt = PromptTemplate.from_template(
    """Analyze the following note content and provide a summary, tags, and sentiment analysis.
Return the result as a JSON object with the following fields:
- summary: A concise one-paragraph summary.
- tags: A list of relevant keywords (max 5).
- sentiment: A string indicating sentiment and confidence, e.g., "Positive (100%)".

Note Title: {title}
Note Content: {content}

Output JSON:"""
)
