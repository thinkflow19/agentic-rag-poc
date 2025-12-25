"""Core agent creation and execution logic."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Callable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents.factory import create_agent

from runtime.tools.document_search import document_search
from runtime.tools.internet_search import internet_search

# Suppress tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_agent_prompt() -> str:
    """Load agent prompt from file."""
    prompt_path = Path(__file__).parent / "prompts" / "agent_prompt.txt"
    with open(prompt_path, 'r') as f:
        return f.read()


def create_agent_executor() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create and configure the LangChain native ReAct agent.
    
    Returns:
        Agent executor function that takes input_dict and returns output_dict
    """
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please set it in .env file or environment variables."
        )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
    
    # Define tools
    tools = [document_search, internet_search]
    
    # Load system prompt
    system_prompt = load_agent_prompt()
    
    # Create LangChain native agent (returns LangGraph state graph)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    # Create executor function that invokes the agent graph
    def agent_executor(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent using LangChain's native ReAct pattern.
        
        Args:
            input_dict: Dictionary with 'input' (query) and optional 'chat_history'
            
        Returns:
            Dictionary with 'output' (agent response)
        """
        from langchain_core.messages import HumanMessage, AIMessage
        
        user_input = input_dict["input"]
        chat_history = input_dict.get("chat_history", [])
        
        # Build messages list from chat history + new user input
        conversation_messages = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:  # Skip empty messages
                continue
            if role == "user":
                conversation_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                conversation_messages.append(AIMessage(content=content))
            # Skip unknown message types
        
        # Add current user input
        conversation_messages.append(HumanMessage(content=user_input))
        
        # LangGraph agents work with message objects
        # Invoke the agent graph with conversation history
        result = agent.invoke({"messages": conversation_messages})
        
        # Extract the final response from LangGraph state
        # The state contains "messages" list with the conversation
        if isinstance(result, dict) and "messages" in result:
            result_messages = result["messages"]
            if result_messages:
                # Get the last message (should be the final answer)
                last_message = result_messages[-1]
                # Handle both dict and message object formats
                if isinstance(last_message, dict):
                    content = last_message.get("content", "")
                elif hasattr(last_message, 'content'):
                    content = last_message.content
                else:
                    content = str(last_message)
                
                return {"output": content}
        
        # Fallback if structure is different
        return {"output": str(result)}
    
    return agent_executor
