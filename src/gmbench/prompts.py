"""Reader and judge prompts, reproduced from the RAW-RAG paper.

The templates below are the ones in appendices G.3 and G.4 of arXiv:2610.30755v1
("Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We
Thought"). They are kept verbatim on purpose: holding the reader and the judge
fixed is what makes a score comparable to the paper's, so that any difference
is attributable to retrieval rather than to prompting.
"""

from __future__ import annotations

LOCOMO_READER = """You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory 1: [timestamp] - [content]
- Memory 2: [timestamp] - [content]
...

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown → Memory 2 mentions A's hometown is LA → Therefore A moved from LA"]
- Inferred facts: [list any facts that require combining information from multiple memories]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references]
- Original reference: [e.g., "last year" from May 2022]
- Calculated actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION CHECK
[If multiple memories contain different information]
- Conflicting information: [describe]
- Resolution: [explain which is most recent/reliable]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included: [list them]
- [ ] All locations included: [list them]
- [ ] All numbers exact: [list them]
- [ ] All frequencies specific: [list them]
- [ ] All dates/times precise: [list them]
- [ ] All proper nouns preserved: [list them]

## STEP 7: ANSWER FORMULATION
[Explain how you're combining the information to answer the question]

## FINAL ANSWER:
[Provide the concise answer with ALL specific details preserved]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:"""

JUDGE_SYSTEM = (
    "You are an expert grader that determines if answers to questions match a gold "
    "standard answer"
)

JUDGE_USER = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


# MemoryAgentBench, appendix G.7. The reader takes the retrieved memories as a
# system message and the source-specific task template as the user message. The
# fact-consolidation template is reproduced verbatim, including its illustrative
# answer, which is part of the original synthetic-knowledge instruction and
# matters: it is what tells the reader to answer from the knowledge pool rather
# than from the world.
MAB_READER_SYSTEM = """You are a helpful AI. Answer the question based on query and memories.
{context}"""

MAB_READER_USER = """{task_query}

Current Time: {current_time}"""

MAB_FACT_CONSOLIDATION_QUERY = """Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number.
You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world.

For example:

[Knowledge Pool]

Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia?
Answer: Donald Trump

Now Answer the Question: Based on the provided Knowledge Pool, {question}

Answer:"""

MAB_RULER_QA_QUERY = """Answer the question based on the memorized documents. Only give me the answer and do not output any other words.

Now Answer the Question: {question}"""

MAB_LONGMEMEVAL_QUERY = """The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.

{question}

Answer:"""

MAB_EVENTQA_QUERY = """Based on the context you memorized, complete the task below:

{question}

The event that happens next is:"""




MAB_ICL_QUERY = """Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else.

Question:{question}

label:"""

MAB_REDIAL_QUERY = """Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences.

For Example:

[Conversation]

The recommendations are:
1.movie1
2.movie2
...

Here is the conversation: {question}

The recommendations are:"""

MAB_DETECTIVE_QA_QUERY = """Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.

{question}"""

MAB_INFBENCH_SUM_QUERY = """You are given a book above and you are tasked to summarize it.

{question}

Now summarize the book."""

# InfBench summaries cannot be graded by string containment: the task is a
# 1,000-word plot summary and the benchmark ships per-question keypoints as its
# rubric. This asks a judge how many of those keypoints the summary actually
# covers, which is what "summary scoring" means for this source.
MAB_SUMMARY_JUDGE_SYSTEM = (
    "You are an expert grader assessing whether a book summary covers the "
    "required plot points. Judge only coverage of the listed key points, not "
    "style, length or writing quality."
)

MAB_SUMMARY_JUDGE_USER = """Below is a summary of a book, followed by the key plot points it was required to cover.

For each key point, decide whether the summary conveys it. A point counts as covered if the summary states the same substance, even in different words. It does not count if the summary omits it, contradicts it, or is too vague to tell.

SUMMARY:
{summary}

KEY POINTS:
{keypoints}

Return only a JSON object with the key "covered", whose value is the number of key points the summary covers, and the key "total", whose value is the number of key points listed. No other text."""


# Which task template each source uses. This must cover every source in
# memoryagentbench.GRADERS; a missing entry is caught before a run starts
# rather than after a few hundred failed questions.
MAB_QUERIES: dict[str, str] = {
    "eventqa": MAB_EVENTQA_QUERY,
    "longmemeval": MAB_LONGMEMEVAL_QUERY,
    "ruler_qa1": MAB_RULER_QA_QUERY,
    "ruler_qa2": MAB_RULER_QA_QUERY,
    "factconsolidation_sh": MAB_FACT_CONSOLIDATION_QUERY,
    "factconsolidation_mh": MAB_FACT_CONSOLIDATION_QUERY,
    "icl_banking77": MAB_ICL_QUERY,
    "icl_clinic150": MAB_ICL_QUERY,
    "icl_nlu": MAB_ICL_QUERY,
    "icl_trec_coarse": MAB_ICL_QUERY,
    "icl_trec_fine": MAB_ICL_QUERY,
    "recsys_redial": MAB_REDIAL_QUERY,
    "detective_qa": MAB_DETECTIVE_QA_QUERY,
    "infbench_sum": MAB_INFBENCH_SUM_QUERY,
}
