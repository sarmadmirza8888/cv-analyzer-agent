from cv_agent.output_cleaning import clean_react_output


def test_trims_invented_next_turn():
    assert clean_react_output("100\n\nQuestion:") == "100"


def test_cuts_at_earliest_marker_even_with_text_after():
    raw = "Ali knows Python.\nThought: I should keep going\nHuman: and then what?"
    assert clean_react_output(raw) == "Ali knows Python."


def test_clean_answer_is_unchanged():
    assert clean_react_output("  Plain answer.  ") == "Plain answer."


def test_empty_input_passes_through():
    assert clean_react_output("") == ""
    assert clean_react_output(None) is None
