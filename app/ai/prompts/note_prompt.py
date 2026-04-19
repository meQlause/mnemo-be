from langchain_core.prompts import PromptTemplate

no_context_prompt = PromptTemplate.from_template(
    """You are a helpful assistant with access to the user's personal notes.

The user has asked a question, but NO relevant information was found in their notes.

## Rules
1. You MUST NOT use external knowledge.
2. You MUST NOT answer the question.
3. You MUST clearly state that no relevant context was found.
4. You MUST remind the user that you are restricted to their notes.

## Output Format Example (be creative but concise)

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
Answer naturally and conversationally in the same language as the question.

Use the notes as your primary source. You may use general knowledge only to clarify or expand.
Keep your answer concise and coherent — do not cut off mid-sentence.

## Notes Context
{context}

## Conversation History
{history}

## Question
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

Before extracting any date, identify what the text is *primarily about*. Work through these questions in order:

1. What is the central subject or main narrative arc of this text?
2. Which event receives the most descriptive attention, detail, or explanation?
3. Is any event only present because it delivered, introduced, or framed another event?
4. Is any event explicitly marked as background, contrast, habit, or supporting example?

**Disqualify** any date expression attached to an event that is:
- A framing or delivery event whose sole narrative function is to introduce another event ("she called to tell me about X", "he wrote to say that Y happened", "I read that Z occurred") — the delivered/introduced event is primary, not the delivery mechanism
- Introduced as contrast or illustration ("on another day", "unlike last time", "for example", "such as")
- A single-sentence aside within a longer narrative about a different time period
- Clearly subordinate to a dominant theme that has its own time reference
- A recurring habit or routine rather than a specific one-time event — prefer the one-time specific event
- Inside a hypothetical, conditional, or counterfactual clause ("if it had happened...", "had she arrived...", "imagine next week...", "would have been...")
- Inside reported speech or a quoted message where the temporal anchor belongs to the speaker's past frame, not the text's present ("she said 'next week the results will be ready'" — 'next week' is relative to when she spoke, not to {reference_date})
- A cancelled, rescheduled, or planned-but-not-executed event unless the text is explicitly about the plan itself
- Negated ("it did NOT happen last month" — do not extract last month)

## STEP 2: ESTABLISH THE TEMPORAL REFERENCE FRAME

Before resolving any date, determine what it is anchored to.

**Rule A — Default anchor:** All relative expressions resolve against {reference_date} UNLESS an explicit date anchor appears earlier in the text and the expression is grammatically chained to it.

**Rule B — Chained anchor:** If an offset expression follows a stated date, resolve against that stated date, not {reference_date}.
> Example: "She was born in 1990. Six years later she started school." → 1990 + 6 = 1996, not {reference_date} − 6 years.

**Rule C — Negation resolution:** When a date expression is negated, identify the actual implied date from the correction and resolve that instead.
> Example: "not last month — it was the month before that" → resolve to 2 months before {reference_date}, not 1 month.

**Rule D — Reported speech anchor shift:** Dates inside quotes or reported speech are relative to the original speaker's moment, which may be unknown or different from {reference_date}. If that speaker's moment cannot be determined from context, the date is unresolvable — return null for it. Do not resolve it against {reference_date}.

**Rule E — Uncertainty spanning:** If the text explicitly states uncertainty between two possible dates or offsets (e.g. "three or four summers later"), resolve both and return the full spanning range. Set confidence to LOW and note the ambiguity in reasoning.

## STEP 3: EXTRACTION PRIORITY

Apply this priority only *within* the primary event identified in Step 1, after anchoring is established in Step 2:

1. Explicit absolute dates (e.g. "June 5", "2023-08-18", "March 3rd", "the 18th of August 2019")
2. Explicit relative dates anchored to {reference_date} (e.g. "yesterday", "last Tuesday", "3 weeks ago")
3. Period references tied to the main event (e.g. "last year", "this month", "last summer", "early 2022")
4. Chained offsets anchored to a stated date in the text (e.g. "six years later", "forty-seven days into the journey")
5. Seasonal or vague period references (LOW confidence only)

**Tiebreaker when two expressions are equally bound to the primary event:**
- More specific beats less specific (a day beats a month; a month beats a year)
- If equally specific, take the first-mentioned and set confidence to MEDIUM
- If both are needed to describe the same event's range, combine into a single range

## STEP 4: RESOLUTION RULES

**Single-day**
- today / just now / earlier today → {reference_date}
- yesterday → -1d
- tomorrow → +1d
- day before yesterday → -2d
- day after tomorrow → +2d

**Weeks**
- this week → start of current ISO week (Monday)
- last week / this past week → start of previous ISO week (Monday)
- next week → start of next ISO week (Monday)
- a week ago / one week ago → -7d from {reference_date}
- in a week → +7d from {reference_date}

**Months**
- this month → first day of current month
- last month → first day of previous month
- next month → first day of next month
- a month ago → same calendar day, previous month
- in a month → same calendar day, next month

**Years**
- this year → YYYY-01-01 (current year)
- last year → (YYYY-1)-01-01
- next year → (YYYY+1)-01-01
- a year ago → same month/day, previous year
- in a year → same month/day, next year

**Generalized offset**
- "N days/weeks/months/years ago" → subtract N units from {reference_date}
- "in N days/weeks/months/years" → add N units to {reference_date}
- "N days/weeks/months/years later" (chained) → add N units to the stated anchor date, not {reference_date}

**Weekdays**
- "last [Day]" → most recent [Day] strictly before {reference_date}
- "next [Day]" → next [Day] strictly after {reference_date}
- "this [Day]" → [Day] within the current ISO week

**Ordinal weekday-in-month**
- "the first/second/third/fourth/last [Day] of [Month]" → compute the exact calendar date; assume current year if no year is stated

**Seasons (Northern Hemisphere default)**
- last summer → previous YYYY-06-01/YYYY-08-31
- this summer → current YYYY-06-01/YYYY-08-31
- last winter → previous YYYY-12-01/YYYY-02-28 (spanning Dec–Feb)
- this winter → current YYYY-12-01/YYYY-02-28
- (Use MEDIUM confidence; note Southern Hemisphere possibility if context implies it)

**Sub-period qualifiers**
- early [month/season/year] → first ~one-third of that period as a range
- mid [month/season/year] → middle ~one-third of that period as a range
- late [month/season/year] → final ~one-third of that period as a range
- Always output as YYYY-MM-DD/YYYY-MM-DD range, never a single day

**Fiscal / academic periods** (LOW confidence — note ambiguity in reasoning)
- last quarter → first day of previous calendar quarter (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
- this quarter → first day of current calendar quarter
- last semester / last term → approximately -6 months from {reference_date}, first of that month

**Duration vs point-in-time**
- "for N years/months/weeks" → duration descriptor, NOT a date; do not extract unless a start or end anchor is explicit in the text
- "over the past N [units]" → range: (reference_date − N units) / {reference_date}
- "N years/months ago" → point-in-time; extract and resolve normally

**Ambiguous vague expressions**
- recently / lately → ({reference_date} − 14d) / {reference_date} — LOW confidence
- a while ago / some time ago → ({reference_date} − 21d) / ({reference_date} − 7d) — LOW confidence
- back then / at that time → resolve only if an explicit referent date appears earlier in the text; otherwise unresolvable

## STEP 5: PARTIAL DATE INFERENCE

- Day only ("the 5th") → assume current month + year; if that date is future relative to {reference_date}, assume previous month
- Day + Month ("August 18") → assume current year; if that date is future relative to {reference_date}, assume previous year
- Month only ("in August") → first day of that month, current year; same future-correction applies
- **Exception:** if the text contains an explicit year anchor ("this year", "in 2025", etc.), that anchor overrides the future-correction rule entirely

## CONSTRAINTS

- NEVER return {reference_date} unless the text explicitly says "today" or "just now"
- NEVER extract a date from background context, habit, routine, or emotional framing
- NEVER extract a date from inside a hypothetical, conditional, counterfactual, or reported-speech clause unless the anchor moment of that speech is explicitly known
- NEVER extract a date from a negated expression — resolve the corrected/actual date instead
- NEVER treat a duration phrase ("for N years") as a point-in-time date unless an explicit anchor makes the start or end unambiguous
- NEVER let a higher-priority date expression override a lower-priority one if the higher-priority expression is attached to a framing event, subordinate clause, contrast, cancelled plan, or a different character's timeline
- NEVER fabricate a date not directly inferable from the text
- If two equally valid expressions remain after all filtering, take the more specific one; if equally specific, take the first-mentioned and set confidence to MEDIUM
- If the text states explicit uncertainty spanning two possible dates, return the full range and set confidence to LOW
- If no resolvable date exists after all steps, return null in event_date with a clear reason in event_reasoning

## OUTPUT

Return strict JSON only. No markdown fences. No extra keys. No preamble.

{{
  "event_date": "<YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD for ranges, or null>",
  "event_confidence": "HIGH | MEDIUM | LOW",
  "event_reasoning": "<one sentence: name the primary event, name the chosen date expression, state why it was chosen, and name any competing expressions that were disqualified and why>"
}}

Text:
{input}
"""
)
