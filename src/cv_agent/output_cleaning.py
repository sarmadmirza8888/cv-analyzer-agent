"""Trim text a small local model generates after it has already answered."""


def clean_react_output(text: str) -> str:
    """
    The local Mistral model sometimes keeps generating after it has already
    given the real answer, and starts hallucinating a brand-new
    Question/Thought/Human turn on its own (visible in earlier runs as
    stray "Question:" text tacked onto the end of an answer, e.g.
    "Final Answer: 100\n\nQuestion:").

    If that leaked, half-formed turn gets returned as a tool's output, the
    calling agent treats it as real context and runs off on an unrelated,
    invented task (this is the root cause of the hallucination where the
    main agent starts answering questions nobody asked). This trims any
    output back to just the real answer by cutting at the first sign the
    model started a new turn by itself.
    """
    if not text:
        return text

    cut_markers = ["\nQuestion:", "\nThought:", "\nHuman:", "\nAI:", "\nObservation:"]
    for marker in cut_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    return text.strip()
