"""
Intent Classification Operator (Logical Operator)

Classifies intent type based on user input: chat or reasoning.
"""
from agent.shared.state import AgentState
from llm.qwen_wrapper import QwenWrapper


def classify_intent(user_text: str) -> str:
    """
    Classify user intent
    
    :param user_text: User input text
    :return: "chat" or "reasoning"
    """
    qwen = QwenWrapper()  # Create instance to call Qwen model, create new instance for each call
    messages = [  # System prompt
        {
            "role": "system",
            "content": (
                "You are an intent classification assistant, you can only choose one of two labels:\n"
                "1. chat: indicates the user just wants to chat or ask questions\n"
                "2. reasoning: indicates the user uploaded an image and wants to analyze the shooting location, identify location, or reason about geographic information\n"
                "Please strictly reply with only one label: chat or reasoning, do not add any other content."
            )
        },
        {
            "role": "user",
            "content": f"User input is as follows:\n{user_text}\nPlease determine the user's intent type:"
        }
    ]
    intent = qwen.chat(messages).strip().lower()  # LLM-Qwen gets user input interaction form
    if intent not in {"chat", "reasoning"}:  # Other options default to chat
        intent = "chat"
    return intent


def classify_intent_operator(state: AgentState) -> AgentState:
    """
    LangGraph node function: Intent classification
    
    :param state: Agent state
    :return: Updated state (contains intent field)
    """
    user_input = state.get("user_text", "")  # Get user input string (empty if no input)
    intent = classify_intent(user_input)
    new_state = state.copy()  # Create new state, new state intent is intent, return new state
    new_state["intent"] = intent
    return new_state


def route_intent_condition(state: AgentState) -> str:
    """
    Routing condition function: Determine next step based on intent
    
    :param state: Agent state
    :return: "chat" or "reasoning"
    """
    return state.get("intent")

