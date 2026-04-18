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
    Output only the raw text formatting. DO NOT output any HTML tags (like <p>, <br>, etc.). Do not include a title.
    """
)

extract_event_date_prompt = PromptTemplate.from_template(
    """Extract the main event date from the text.

## Core Rules
1. Select ONLY the time expression directly tied to the main event/action.
2. IGNORE time expressions used for background, description, or context.
3. If multiple candidates exist, choose the one most strongly linked to the main event.

## Reference Date Usage (CRITICAL)
- The reference date ({reference_date}) is ONLY a base for calculation.
- NEVER return the reference date as the event date unless the user explicitly says "today".
- For relative phrases, resolve them STERNLY:
    - "yesterday" → {reference_date} minus 1 day
    - "this past week" → {reference_date} minus 7 days (the start of the reflection period)
    - "last week" → {reference_date} minus 7 days
    - "last month" → {reference_date} minus 1 month

## Date Inference
- If ONLY day is mentioned (e.g. "on the 18th"), assume current month and year from reference date.
- If day + month is mentioned (e.g. "August 18"), assume current year from reference date.

Reference date: {reference_date}

Text:
{input}
"""
)
