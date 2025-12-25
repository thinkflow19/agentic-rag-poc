"""Core agent creation and execution logic."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Callable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents.factory import create_agent

from app.core.tools.document_search import document_search
from app.core.tools.internet_search import internet_search

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv()

logger = logging.getLogger(__name__)


def load_agent_prompt() -> str:
    """
    Load agent prompt from file.
    
    Returns:
        Prompt text as string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "agent_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Agent prompt not found: {prompt_path}")
    with open(prompt_path, 'r') as f:
        return f.read()


def create_agent_executor() -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create and configure the LangChain ReAct agent.
    
    Returns:
        Agent executor function that takes input_dict and returns output_dict.
        Input dict should have 'input' (query string) and optional 'chat_history'.
        Output dict contains 'output' (agent response string).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please set it in .env file or environment variables."
        )
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
    
    tools = [document_search, internet_search]
    system_prompt = load_agent_prompt()
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    def agent_executor(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        from langchain_core.messages import HumanMessage, AIMessage
        
        user_input = input_dict["input"]
        chat_history = input_dict.get("chat_history", [])
        
        conversation_messages = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                conversation_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                conversation_messages.append(AIMessage(content=content))
        
        conversation_messages.append(HumanMessage(content=user_input))
        
        result = agent.invoke({"messages": conversation_messages})
        
        if isinstance(result, dict) and "messages" in result:
            result_messages = result["messages"]
            if result_messages:
                last_message = result_messages[-1]
                if isinstance(last_message, dict):
                    content = last_message.get("content", "")
                elif hasattr(last_message, 'content'):
                    content = last_message.content
                else:
                    content = str(last_message)
                
                return {"output": content}
        
        return {"output": str(result)}
    
    return agent_executor
