"""Evaluation script for testing agent queries."""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents.factory import create_agent
from langchain_core.messages import HumanMessage

from app.core.tools.document_search import document_search
from app.core.tools.internet_search import internet_search

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_agent_prompt() -> str:
    """
    Load agent prompt from file.
    
    Returns:
        Prompt text as string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = Path(__file__).parent.parent / "app" / "core" / "prompts" / "agent_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Agent prompt not found: {prompt_path}")
    with open(prompt_path, 'r') as f:
        return f.read()


def create_agent():
    """
    Create agent for evaluation.
    
    Returns:
        Agent executor function that takes input_dict and returns output_dict
        
    Raises:
        ValueError: If OPENAI_API_KEY is not found
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    tools = [document_search, internet_search]
    
    system_prompt = load_agent_prompt()
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    def executor(input_dict):
        result = agent.invoke({
            "messages": [HumanMessage(content=input_dict["input"])]
        })
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    return {"output": last_message.get("content", "")}
                elif hasattr(last_message, 'content'):
                    return {"output": last_message.content}
        return {"output": str(result)}
    
    return executor


def run_test_queries():
    """
    Run predefined test queries and print results.
    Tests include identifier lookup, location queries, summarization, and web fallback.
    """
    test_queries = [
        "Find contract id HD-7961",
        "Where does HD-7961 appear?",
        "Summarize the certificate of insurance form",
        "What is BM25?"
    ]
    
    print("\n" + "="*80)
    print("EVALUATION: Running Test Queries")
    print("="*80 + "\n")
    
    agent = create_agent()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}: {query}")
        print(f"{'='*80}\n")
        
        try:
            result = agent({"input": query})
            
            print("TOOL USAGE:")
            print("- Check agent logs above for tool calls")
            print("\nFINAL ANSWER:")
            print(result["output"])
            print("\n" + "-"*80)
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            print(f"ERROR: {e}\n")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80 + "\n")


def main():
    """
    Main entry point for evaluation script.
    Can run predefined test suite or a single query via --query argument.
    """
    parser = argparse.ArgumentParser(description='Test agent queries')
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to test (optional)'
    )
    
    args = parser.parse_args()
    
    if args.query:
        agent = create_agent()
        result = agent({"input": args.query})
        print("\n" + "="*80)
        print("QUERY:", args.query)
        print("="*80)
        print("\nANSWER:")
        print(result["output"])
        print("="*80)
    else:
        # Run test suite
        run_test_queries()


if __name__ == '__main__':
    main()

