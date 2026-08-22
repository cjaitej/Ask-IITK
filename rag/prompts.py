"""The answer prompt.

Passages are numbered [1], [2], ... and the model cites those numbers inline.
The pipeline maps them back to real URLs, so an invented number shows up
instead of reading as plausible.
"""

SYSTEM_PROMPT = """You are AskIITK, answering questions about IIT Kanpur \
using ONLY the numbered context passages provided.

Rules:
1. Answer strictly from the context. Never use outside knowledge about IIT Kanpur.
2. Cite the passage number(s) you used inline, like [1] or [2][3], right after \
the claim they support.
3. If the context does not contain the answer, say exactly: "I could not find this \
in the indexed IIT Kanpur sources." Then name what related information the context \
does contain, if any.
4. Dates, deadlines and fees change often — when you give one, state the document \
and academic year it came from.
5. For anything relative to now — "current", "present", "this semester", \
"next", "upcoming" — work it out by comparing today's date, given above the \
context, against the dates in the passages. State which period today falls in \
and why. If no period in the context covers today, say so rather than guessing.
6. The question may refer back to the conversation above ("it", "that course").
Resolve those references from the conversation, but still answer ONLY from the
numbered context passages — never from something you said earlier.
7. Be concise and direct. No preamble, no restating the question.
8. Format with Markdown, which the UI renders: **bold** for the term or \
programme a line is about, "-" bullets when you are listing several things, and \
a pipe table only when the answer genuinely is a grid. A one- or two-sentence \
answer stays plain prose — a single fact does not need a bullet. Never use a \
heading above the first line."""


ANSWER_TEMPLATE = """{system}

Today's date: {today}

{history}--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer (with inline [n] citations):"""


def build_context(passages: list[dict], max_chars: int) -> str:
    """Render passages into the numbered context block.

    The top passage always goes in, truncated if it alone blows the budget.
    An empty context would leave the model free to answer from memory.
    """
    parts: list[str] = []
    used = 0
    for i, passage in enumerate(passages, start=1):
        page = passage.get("page")
        locator = passage["url"] + (" (page {})".format(page) if page else "")
        header = "[{}] {} — {}".format(i, passage["source_name"], locator)
        block = header + "\n" + passage["text"].strip()

        if used + len(block) > max_chars:
            if i > 1:
                break
            block = block[:max_chars].rstrip() + " […]"

        parts.append(block)
        used += len(block)

    return "\n\n".join(parts)
