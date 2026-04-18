from langchain_core.prompts import PromptTemplate

no_context_prompt = PromptTemplate.from_template(
    """You are a helpful assistant with access to the user's personal notes.

The user has asked a question, but NO relevant information was found in their notes.

## Rules
1. You MUST NOT use external knowledge.
2. You MUST NOT answer the question.
3. You MUST clearly state that no relevant context was found.
4. You MUST remind the user that you are restricted to their notes.

## Output Format (STRICT)

> I don't have enough context in your personal notes to answer this question.

### Explanation
* The requested information is not available in the provided notes
* I am restricted to using only your personal notes

## Conversation History
{history}

## Question
{input}
"""
)

rag_initial_prompt = PromptTemplate.from_template(
    """You are a helpful assistant answering questions based ONLY on the user's personal notes.

## Rules
1. Use ONLY the provided **Context from Personal Notes**.
2. DO NOT use any external knowledge.
3. If the answer is not in the context, respond with:
   > "I don't have enough context in the provided notes."
4. Be concise and professional.

## Output Format (STRICT)

<direct answer based only on context>

### Key Points
* <point 1>
* <point 2>

## Formatting Rules
- Use **bold** for important terms
- Use *italics* for emphasis
- Use backticks for code or technical terms: `example`
- Use bullet points with `*` ONLY (no `-`)
- Always return valid Markdown
- Do NOT return plain text

## Context from Personal Notes
{context}

## Conversation History
{history}

## Question
{input}
"""
)

rag_followup_prompt = PromptTemplate.from_template(
    """You are a helpful assistant discussing the user's personal notes.

A context has already been established.

## Rules
1. Base your answer primarily on:
   - **Context from Personal Notes**
   - **Conversation History**
2. You MAY use general knowledge ONLY to:
   - clarify
   - expand
   - explain implications
3. You MUST stay consistent with the notes.
4. If the question is unrelated, gently steer back to the notes.

## Output Format (STRICT)

<clear and helpful response>

### Key Points
* <point 1>
* <point 2>

## Formatting Rules
- Use **bold** for important terms
- Use *italics* for emphasis
- Use backticks for code or technical terms: `example`
- Use bullet points with `*` ONLY
- Always return valid Markdown
- Do NOT return plain text

## Context from Personal Notes
{context}

## Conversation History
{history}

## Question
{input}
"""
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

random_note_prompt = PromptTemplate.from_template(
    """You are a creative writer. Generate a completely random, interesting, and coherent note consisting of exactly 2 paragraphs.
    The topic could be anything: science, history, casual thoughts, a short fictional story, or interesting facts.
    Output only the note text. Do not include a title.
    """
)
