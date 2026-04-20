from langchain_core.prompts import PromptTemplate

no_context_prompt = PromptTemplate.from_template(
    """You are a helpful assistant with access to the user's personal notes.
The user has asked a question but no relevant information was found in their notes.
Respond in one plain sentence only. No headers, no bullet points, no markdown formatting of any kind.
Do not explain why. Just say you don't have that information in their notes.

Conversation History:
{history}

Question:
{input}
"""
)

rag_initial_prompt = PromptTemplate.from_template(
    """You are a helpful assistant answering questions based ONLY on the user's personal notes.
Write naturally in second person as if speaking directly to the person who wrote the notes.
Sound like a knowledgeable friend summarizing their own notes back to them, not a formatted report.

Rules:
- Use ONLY the provided context from personal notes
- Never copy-paste from the context, always paraphrase in your own words
- Never use external knowledge
- Write complete grammatically correct sentences
- If the answer is not in the context say in one plain sentence: I don't have enough information in your notes to answer this

Do not use any Markdown formatting whatsoever. No asterisks, no pound signs, no backticks, no bullet symbols, no headers of any kind. Plain sentences only.
Write like a human would speak. One concise paragraph only.

When quoting exact terms or phrases directly from the notes, wrap them in double quotes like "critical fatigue crack".
Use bold only for the single most important term or finding in the response, not for entire phrases or dates.

If there are multiple distinct facts worth mentioning, weave them naturally into the prose rather than listing them.

Context from Personal Notes:
{context}

Conversation History:
{history}

Question:
{input}
"""
)

rag_followup_prompt = PromptTemplate.from_template(
    """You are a helpful assistant discussing the user's personal notes.
Answer strictly in the same language as the Question field below, regardless of the language in the notes context.

Use the notes as your primary source. If the question asks for recommendations, opinions, or general knowledge that goes beyond what is in the notes, you may answer freely from your own knowledge — just do not contradict anything stated in the notes.

Do not use any Markdown formatting whatsoever. No asterisks, no pound signs, no backticks, no bullet symbols, no headers of any kind. Plain sentences only.
Write one concise paragraph only. Do not cut off mid-sentence.
If the answer involves multiple options or choices, pick the single most relevant one and answer as if it were the only option. Do not list alternatives.
If the context does not contain relevant information and the question cannot be answered from general knowledge either, say so in one plain sentence only.

Notes Context:
{context}

Conversation History:
{history}

Question:
{input}
"""
)

title_prompt = PromptTemplate.from_template(
    """Given the following text, generate a short, concise title (max 5 words).
Do not include quotes or any other formatting. Output only the title, nothing else.

Text:
{text}

Title:"""
)

analyze_prompt = PromptTemplate.from_template(
    """Analyze the following note content and provide a summary, tags, and sentiment analysis.
Return the result as a JSON object with exactly these fields:
- summary: a concise one-paragraph summary
- tags: a list of relevant keywords, maximum 5 items
- sentiment: a string indicating sentiment and confidence, e.g. "Positive (100%)"

Return strict JSON only. No markdown fences. No extra keys. No preamble.

Note Title: {title}
Note Content: {content}

Output JSON:"""
)

random_note_prompt = PromptTemplate.from_template(
    """You are a creative writer. Generate a completely random, interesting, and coherent note consisting of exactly 2 paragraphs.
The topic could be anything: science, history, casual thoughts, a short fictional story, or interesting facts.
Output only plain text. No HTML tags, no Markdown, no special formatting of any kind. Do not include a title.
"""
)

extract_event_date_prompt = PromptTemplate.from_template(
    """You are a precise date extraction engine. Extract the ONE date expression most directly tied to the main event or action in the text, then resolve it using the reference date.

<reference_date>{reference_date}</reference_date>

STEP 1 - IDENTIFY THE PRIMARY EVENT

Before extracting any date, identify what the text is primarily about. Work through these questions in order:
1. What is the central subject or main narrative arc of this text?
2. Which event receives the most descriptive attention, detail, or explanation?
3. Is any event only present because it delivered, introduced, or framed another event?
4. Is any event explicitly marked as background, contrast, habit, or supporting example?

Disqualify any date expression attached to an event that is:
- A framing or delivery event whose sole narrative function is to introduce another event ("she called to tell me about X", "he wrote to say that Y happened") — the delivered event is primary, not the delivery mechanism
- Introduced as contrast or illustration ("on another day", "unlike last time", "for example", "such as")
- A single-sentence aside within a longer narrative about a different time period
- Clearly subordinate to a dominant theme that has its own time reference
- A recurring habit or routine rather than a specific one-time event — prefer the one-time specific event
- Inside a hypothetical, conditional, or counterfactual clause ("if it had happened...", "had she arrived...", "imagine next week...", "would have been...")
- Inside reported speech or a quoted message where the temporal anchor belongs to the speaker's past frame, not the text's present
- A cancelled, rescheduled, or planned-but-not-executed event unless the text is explicitly about the plan itself
- Negated ("it did NOT happen last month" — do not extract last month; resolve the corrected date instead)

STEP 2 - ESTABLISH THE TEMPORAL REFERENCE FRAME

Before resolving any date, determine what it is anchored to.

Rule A - Default anchor: All relative expressions resolve against {reference_date} UNLESS an explicit date anchor appears earlier in the text and the expression is grammatically chained to it.

Rule B - Chained anchor: If an offset expression follows a stated date, resolve against that stated date, not {reference_date}.
Example: "She was born in 1990. Six years later she started school." resolves to 1996, not {reference_date} minus 6 years.

Rule C - Negation resolution: When a date expression is negated, identify the actual implied date from the correction and resolve that instead.
Example: "not last month, it was the month before that" resolves to 2 months before {reference_date}.

Rule D - Reported speech anchor shift: Dates inside quotes or reported speech are relative to the original speaker's moment. If the speaker's moment can be determined from context, resolve against it. If it cannot be determined, the date is unresolvable — return null.
Example: "she said next Friday" and the text establishes she spoke last Thursday — resolve next Friday against last Thursday, not against {reference_date}.

Rule E - Uncertainty spanning: If the text explicitly states uncertainty between two possible dates (e.g. "three or four summers later"), resolve both and return the full spanning range. Set confidence to LOW.

Rule F - Composite expressions: When the primary event's date is stated as an offset FROM another stated date, the event date is the result of applying that offset to the anchor, NOT the anchor itself. Always complete the arithmetic fully.
Example: "signed 90 days after February 14th" — event_date is Feb 14 plus 90 days equals May 15, not Feb 14.
Example: "three weeks before August 1st" — event_date is Aug 1 minus 21 days equals July 11.

STEP 3 - EXTRACTION PRIORITY

Apply this priority only within the primary event identified in Step 1, after anchoring is established in Step 2:
1. Explicit absolute dates (e.g. "June 5", "2023-08-18", "March 3rd")
2. Explicit relative dates anchored to {reference_date} (e.g. "yesterday", "last Tuesday", "3 weeks ago")
3. Period references tied to the main event (e.g. "last year", "this month", "last summer", "early 2022")
4. Chained offsets anchored to a stated date in the text (e.g. "six years later", "90 days after February 14th")
5. Seasonal or vague period references (LOW confidence only)

Tiebreaker: more specific beats less specific. If equally specific, take the first-mentioned and set confidence to MEDIUM. If both are needed to describe the same event range, combine into a single range.

STEP 4 - RESOLUTION RULES

Single-day:
today / just now / earlier today → {reference_date}
yesterday → -1d
tomorrow → +1d
day before yesterday → -2d
day after tomorrow → +2d

Weeks:
this week → start of current ISO week (Monday)
last week / this past week → start of previous ISO week (Monday)
next week → start of next ISO week (Monday)
a week ago / one week ago → -7d from {reference_date}
in a week → +7d from {reference_date}

Months:
this month → first day of current month
last month → first day of previous month
next month → first day of next month
a month ago → same calendar day, previous month
in a month → same calendar day, next month

Years:
this year → YYYY-01-01
last year → (YYYY-1)-01-01
next year → (YYYY+1)-01-01
a year ago → same month and day, previous year
in a year → same month and day, next year

Generalized offset:
"N days/weeks/months/years ago" → subtract N units from {reference_date}
"in N days/weeks/months/years" → add N units to {reference_date}
"N days/weeks/months/years later" (chained) → add N units to the stated anchor date, not {reference_date}

Weekdays:
"last [Day]" → most recent [Day] strictly before {reference_date}
"next [Day]" → next [Day] strictly after {reference_date}
"this [Day]" → [Day] within the current ISO week

Ordinal weekday-in-month:
"the first/second/third/fourth/last [Day] of [Month]" → compute the exact calendar date; assume current year if no year is stated

Seasons (Northern Hemisphere default):
last summer → previous YYYY-06-01/YYYY-08-31
this summer → current YYYY-06-01/YYYY-08-31
last winter → previous YYYY-12-01/YYYY-02-28
this winter → current YYYY-12-01/YYYY-02-28
Use MEDIUM confidence; note Southern Hemisphere possibility if context implies it

Sub-period qualifiers:
early [month/season/year] → first one-third of that period as a range
mid [month/season/year] → middle one-third of that period as a range
late [month/season/year] → final one-third of that period as a range
Always output as YYYY-MM-DD/YYYY-MM-DD range, never a single day

Fiscal / academic periods (LOW confidence — note ambiguity in reasoning):
last quarter → first day of previous calendar quarter (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
this quarter → first day of current calendar quarter
last semester / last term → approximately -6 months from {reference_date}, first of that month

Duration vs point-in-time:
"for N years/months/weeks" → duration descriptor, NOT a date; do not extract unless a start or end anchor is explicit
"over the past N units" → range: ({reference_date} minus N units) / {reference_date}
"N years/months ago" → point-in-time; extract and resolve normally

Ambiguous vague expressions:
recently / lately → ({reference_date} minus 14d) / {reference_date} — LOW confidence
a while ago / some time ago → ({reference_date} minus 21d) / ({reference_date} minus 7d) — LOW confidence
back then / at that time → resolve only if an explicit referent date appears earlier in the text; otherwise unresolvable

STEP 5 - PARTIAL DATE INFERENCE

Day only ("the 5th") → assume current month and year; if that date is future relative to {reference_date}, assume previous month
Day + Month ("August 18") → assume current year; if that date is future relative to {reference_date}, assume previous year
Month only ("in August") → first day of that month, current year; same future-correction applies
Exception: if the text contains an explicit year anchor ("this year", "in 2025", etc.), that anchor overrides the future-correction rule entirely

CONSTRAINTS

- NEVER return {reference_date} unless the text explicitly says "today" or "just now"
- NEVER extract a date from background context, habit, routine, or emotional framing
- NEVER extract a date from inside a hypothetical, conditional, counterfactual, or reported-speech clause unless the anchor moment of that speech is explicitly known and stated in the text
- NEVER extract a negated date expression — resolve the corrected actual date instead
- NEVER treat a duration phrase ("for N years") as a point-in-time date unless an explicit anchor makes the start or end unambiguous
- NEVER return an intermediate anchor date when the event date is expressed as an offset from it — always resolve the full composite expression and return the computed result
- NEVER let a higher-priority date expression override a lower-priority one if the higher-priority expression is attached to a framing/delivery event, subordinate clause, contrast, cancelled plan, negation, or a different character's timeline
- NEVER fabricate a date not directly inferable from the text
- If two equally valid expressions remain after all filtering, take the more specific one; if equally specific, take the first-mentioned and set confidence to MEDIUM
- If the text states explicit uncertainty spanning two possible dates, return the full spanning range and set confidence to LOW
- If no resolvable date exists after all steps, return null in event_date with a clear reason in event_reasoning

OUTPUT

Return strict JSON only. No markdown fences. No extra keys. No preamble.

{{
  "event_date": "<YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD for ranges, or null>",
  "event_confidence": "HIGH | MEDIUM | LOW",
  "event_reasoning": "<one sentence only, maximum 50 words: name the primary event, name the chosen date expression, state why it was chosen, and name any competing expressions that were disqualified and why>"
}}

Text:
{input}
"""
)
