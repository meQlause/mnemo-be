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
    """You are a precise date extraction engine. Extract the ONE date expression most directly tied to the main event or action in the text, then resolve it using the reference date.

<reference_date>{reference_date}</reference_date>

## STEP 1: IDENTIFY THE PRIMARY EVENT
Before extracting any date, identify what the text is *primarily about*. Ask:
- What is the central subject or main narrative arc of this text?
- Which event receives the most descriptive attention or explanation?
- Is any event explicitly framed as background, contrast, or a supporting example?

Disqualify any date expression that is:
- Introduced as a contrast or illustration ("on another day", "for example", "such as", "yesterday's")
- A single-sentence aside within a longer narrative about a different time period
- Clearly subordinate to a dominant theme that has its own time reference

## STEP 2: EXTRACTION PRIORITY
Apply this priority only *within* the primary event identified in Step 1:
1. Explicit absolute dates (e.g. "June 5", "2023-08-18")
2. Explicit relative dates (e.g. "yesterday", "last Tuesday", "3 weeks ago")
3. Period references tied to the main event (e.g. "last year", "this month")
4. Implicit/contextual dates inferred from surrounding context

If multiple time expressions exist, select the one most strongly bound to the PRIMARY subject or action — not background detail, habit, or emotional commentary.

## STEP 3: RESOLUTION RULES

Single-day: today/just now/earlier today → {reference_date} | yesterday → -1d | tomorrow → +1d | day before yesterday → -2d | day after tomorrow → +2d

Weeks: this week → start of current ISO week | last week/this past week → start of previous ISO week | next week → start of next ISO week | a week ago → -7d | in a week → +7d

Months: this month → first day of current month | last month → first day of previous month | next month → first day of next month | a month ago → -1 month | in a month → +1 month

Years: this year → first day of current year | last year → first day of previous year | next year → first day of next year | a year ago → -1 year | in a year → +1 year

Generalized: "{{n}} days/weeks/months/years ago" → -{{n}} unit | "in {{n}} days/weeks/months/years" → +{{n}} unit

Weekdays: "last [Day]" → most recent [Day] before {reference_date} | "next [Day]" → next [Day] after {reference_date} | "this [Day]" → [Day] within current week

Ambiguous: recently/lately → range of last 7–14 days | a while ago → last 1–3 weeks (LOW confidence) | back then/at that time → resolve only if referent is explicit in context

## PARTIAL DATE INFERENCE
- Day only (e.g. "the 18th") → assume current month + year
- Day + Month (e.g. "August 18") → assume current year
- Month only (e.g. "in August") → first day of that month, current year

## CONSTRAINTS
- NEVER return {reference_date} unless text explicitly says "today"
- NEVER extract a date describing background, habit, or emotional context
- NEVER fabricate a date not inferable from the text
- NEVER let a higher-priority date override a lower-priority one if the higher-priority date belongs to a subordinate clause, contrast example, or single-sentence aside within a longer primary narrative
- If no resolvable date exists, return null with a brief reason

## OUTPUT (strict JSON)
```json
{{
  "event_date": "<YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD for ranges>",
  "event_confidence": "HIGH | MEDIUM | LOW",
  "event_reasoning": "<one sentence: why this expression was chosen and why competing expressions were disqualified>"
}}
```

Text:
{input}
"""
)
