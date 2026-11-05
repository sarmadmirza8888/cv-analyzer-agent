"""Specialist ReAct agents (CV, calculator, web) and the coordinator that routes to them."""
from dataclasses import dataclass

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from cv_agent.config import COORDINATOR_MAX_ITERATIONS, SPECIALIST_MAX_ITERATIONS
from cv_agent.grounded_parser import GroundedReActOutputParser
from cv_agent.output_cleaning import clean_react_output
from cv_agent.prompts import (
    CALCULATOR_PROMPT_TEMPLATE,
    CV_AGENT_PROMPT_TEMPLATE,
    MAIN_AGENT_PROMPT_TEMPLATE,
    WEB_AGENT_PROMPT_TEMPLATE,
)
from cv_agent.tools import CALCULATOR_TOOLS, make_cv_search_tool, make_web_search_tool


@dataclass
class Specialists:
    cv: AgentExecutor
    calculator: AgentExecutor
    web: AgentExecutor


def build_react_executor(llm, tools, prompt_template: str, max_iterations: int,
                         verbose: bool = True) -> AgentExecutor:
    """One ReAct agent with the grounded parser, parsing-error recovery and an iteration cap."""
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt, output_parser=GroundedReActOutputParser())
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,             # prints each Thought/Action/Observation - great for debugging
        handle_parsing_errors=True,  # if the model's output doesn't match the format, feed the error back instead of crashing
        max_iterations=max_iterations,
        early_stopping_method="force",  # "generate" is not a valid option in this langchain-classic version; "force" just returns whatever answer it has when max_iterations is hit
    )


def build_specialists(llm, vectorstore, verbose: bool = True) -> Specialists:
    return Specialists(
        cv=build_react_executor(
            llm, [make_cv_search_tool(vectorstore)], CV_AGENT_PROMPT_TEMPLATE,
            SPECIALIST_MAX_ITERATIONS, verbose),
        calculator=build_react_executor(
            llm, CALCULATOR_TOOLS, CALCULATOR_PROMPT_TEMPLATE,
            SPECIALIST_MAX_ITERATIONS, verbose),
        web=build_react_executor(
            llm, [make_web_search_tool()], WEB_AGENT_PROMPT_TEMPLATE,
            SPECIALIST_MAX_ITERATIONS, verbose),
    )


def run_specialist(executor: AgentExecutor, user_query: str) -> str:
    result = executor.invoke({"input": user_query})
    return clean_react_output(result["output"])


def build_agent_tools(specialists: Specialists) -> list:
    """Wrap each specialist as a tool. The docstrings are what the coordinator routes on."""

    @tool
    def cv_search_agent_tool(query: str) -> str:
        """Use this subagent to search the CV database for candidate skills, experience, education, projects, and other CV information."""
        return run_specialist(specialists.cv, query)

    @tool
    def calculator_agent_tool(query: str) -> str:
        """Use this subagent for mathematical calculations and arithmetic operations."""
        return run_specialist(specialists.calculator, query)

    @tool
    def web_search_agent_tool(query: str) -> str:
        """Use this subagent to search the internet for current, recent, or external information."""
        return run_specialist(specialists.web, query)

    return [cv_search_agent_tool, calculator_agent_tool, web_search_agent_tool]


def build_coordinator(llm, specialists: Specialists, verbose: bool = True) -> AgentExecutor:
    return build_react_executor(
        llm,
        build_agent_tools(specialists),
        MAIN_AGENT_PROMPT_TEMPLATE,
        COORDINATOR_MAX_ITERATIONS,  # cap so a confused small model can't loop forever
        verbose,
    )
