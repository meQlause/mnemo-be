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
- Introduced as a contrast or illustration ("on another day", "for example", "such as", "unlike last time")
- A single-sentence aside within a longer narrative about a different time period
- Clearly subordinate to a dominant theme that has its own time reference
- Part of a recurring habit or routine rather than a specific one-time event (prefer the one-time event)
- Inside a hypothetical, conditional, or counterfactual clause ("if it had happened...", "had she arrived yesterday...", "imagine next week...")
- Inside reported speech or a quote where the temporal anchor belongs to the speaker's past frame, not the text's present ("she said 'I'll do it tomorrow'" — tomorrow is relative to the speaker, not {reference_date})

## STEP 2: TEMPORAL REFERENCE FRAME

All relative date expressions must be resolved against {reference_date} UNLESS the expression is clearly anchored to a different stated date in the text.

Example: "He was born in 1990. Two years later he moved abroad." → "two years later" resolves to 1992, not {reference_date} minus 2 years, because it is explicitly anchored to 1990.

If a temporal expression is negated, resolve the actual implied date, not the negated one.
Example: "not last year, but the year before" → resolve to first day of the year 2 years before {reference_date}.

## STEP 3: EXTRACTION PRIORITY

Apply this priority only *within* the primary event identified in Step 1:

1. Explicit absolute dates (e.g. "June 5", "2023-08-18", "March 3rd")
2. Explicit relative dates anchored to reference_date (e.g. "yesterday", "last Tuesday", "3 weeks ago")
3. Period references tied to the main event (e.g. "last year", "this month", "last summer")
4. Implicit/contextual dates inferred from surrounding anchored context
5. Seasonal or vague period references (LOW confidence only)

If multiple time expressions exist at the same priority level and are equally bound to the primary event, select the most specific one (a specific day beats a month, a month beats a year). If specificity is equal, select the earliest-mentioned expression and flag confidence as MEDIUM.

## STEP 4: RESOLUTION RULES

**Single-day**
today / just now / earlier today → {reference_date}
yesterday → -1d
tomorrow → +1d
day before yesterday → -2d
day after tomorrow → +2d

**Weeks**
this week → start of current ISO week (Monday)
last week / this past week → start of previous ISO week
next week → start of next ISO week
a week ago / one week ago → -7d
in a week → +7d

**Months**
this month → first day of current month
last month → first day of previous month
next month → first day of next month
a month ago → -1 month (same day, previous month)
in a month → +1 month

**Years**
this year → first day of current year (YYYY-01-01)
last year → first day of previous year
next year → first day of next year
a year ago → -1 year
in a year → +1 year

**Generalized offset**
"{{n}} days/weeks/months/years ago" → subtract {{n}} units from {reference_date}
"in {{n}} days/weeks/months/years" → add {{n}} units to {reference_date}

**Weekdays**
"last [Day]" → most recent [Day] strictly before {reference_date}
"next [Day]" → next [Day] strictly after {reference_date}
"this [Day]" → [Day] within the current ISO week

**Ordinal weekday-in-month**
"the first/second/third/last [Day] of [Month]" → compute the actual calendar date; assume current year if no year given

**Seasons** (Northern Hemisphere default unless context implies otherwise)
last summer → YYYY-06-01 (previous year if current month is before June, else current year)
this summer → YYYY-06-01 (current year)
last winter → YYYY-12-01 of previous year
this winter → YYYY-12-01 of current year
(Use MEDIUM confidence for all seasonal references; note in reasoning if Southern Hemisphere is implied)

**Sub-period qualifiers**
early [month/season/year] → first day of that period
mid [month/season/year] → 15th of that month, or midpoint of that period
late [month/season/year] → last day of that period
Output as a date range: YYYY-MM-DD/YYYY-MM-DD spanning the approximate sub-period

**Fiscal / academic periods** (use LOW confidence; note ambiguity in reasoning)
last quarter → first day of the previous calendar quarter (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
this quarter → first day of current calendar quarter
last semester / last term → approximate: -6 months from {reference_date}, first of that month

**Duration vs point-in-time disambiguation**
"for three years" → describes duration, NOT a date; do not extract unless a start or end anchor is explicit
"three years ago" → point-in-time; extract and resolve normally
"over the past month" → range: first day of previous month / {reference_date}

**Ambiguous vague expressions**
recently / lately → range: ({reference_date} - 14d) / {reference_date} (LOW confidence)
a while ago / some time ago → range: ({reference_date} - 21d) / ({reference_date} - 7d) (LOW confidence)
back then / at that time → resolve only if an explicit referent date appears earlier in the text; otherwise return null

## PARTIAL DATE INFERENCE

- Day only (e.g. "the 18th") → assume current month + year; if that date is in the future relative to {reference_date}, assume previous month
- Day + Month (e.g. "August 18") → assume current year; if that date is in the future relative to {reference_date}, assume previous year
- Month only (e.g. "in August") → first day of that month, current year; same future-correction rule applies

## CONSTRAINTS

- NEVER return {reference_date} unless text explicitly says "today" or "just now"
- NEVER extract a date describing background, habit, routine, or emotional context
- NEVER extract a date from inside a hypothetical, conditional, or reported-speech clause
- NEVER fabricate a date not directly inferable from the text
- NEVER let a higher-priority date expression override a lower-priority one if the higher-priority expression belongs to a subordinate clause, contrast, aside, or a different character's timeline
- NEVER treat a duration phrase ("for N years") as a point-in-time date unless an explicit anchor makes the start or end unambiguous
- If two equally valid date expressions remain after all filtering, pick the more specific one; if equally specific, pick the first-mentioned and set confidence to MEDIUM
- If no resolvable date exists, return null in event_date with a brief reason in event_reasoning

## OUTPUT (strict JSON — no markdown fences, no extra keys)

{{
  "event_date": "<YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD for ranges, or null>",
  "event_confidence": "HIGH | MEDIUM | LOW",
  "event_reasoning": "<one sentence: the chosen expression, why it was chosen, and why any competing expressions were disqualified>"
}}

Text:
{input}
"""
)