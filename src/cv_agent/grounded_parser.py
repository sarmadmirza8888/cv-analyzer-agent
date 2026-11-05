"""ReAct output parser that keeps the Action when the model also guesses a Final Answer."""
import re

from langchain_classic.agents.output_parsers.react_single_input import (
    FINAL_ANSWER_ACTION,
    ReActSingleInputOutputParser,
)


class GroundedReActOutputParser(ReActSingleInputOutputParser):
    """
    The local model sometimes gets impatient: instead of stopping right
    after "Action Input: ...", it also bolts on a guessed Final Answer in
    the same turn (e.g. "Final Answer: ... are [candidate names]" -- note
    the unfilled placeholder, it never actually looked anything up).

    The stock parser treats "Action + Final Answer in the same turn" as a
    hard error. That error text then gets stitched into the *next* prompt
    as if it were part of the transcript, and the model hallucinates
    further trying to make sense of it (e.g. inventing a fake
    `visit_documentation` tool out of the troubleshooting link in the
    error message).

    Fix: when both appear, trust the Action and silently drop the guessed
    Final Answer, so the real tool actually gets called and the eventual
    answer is grounded in a real Observation instead of a guess.
    """

    def parse(self, text: str):
        if FINAL_ANSWER_ACTION in text:
            action_regex = (
                r"Action\s*\d*\s*:[\s]*(.*?)"
                r"Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            )
            if re.search(action_regex, text, re.DOTALL):
                text = text.split(FINAL_ANSWER_ACTION)[0].strip()
        return super().parse(text)
