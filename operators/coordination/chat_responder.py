"""
Chat Response Operator (Logical Operator)

Uses LLM to respond to user's chat requests.
"""
from agent.shared.state import AgentState
from llm.qwen_wrapper import QwenWrapper


def chat_responder_operator(state: AgentState) -> AgentState:
    """
    LangGraph node function: LLM chat response
    
    :param state: Agent state
    :return: Updated state (contains chat_response)
    """
    user_text = state.get("user_text", "")  # Get user input string (empty if no input)
    qwen = QwenWrapper()  # Create instance to call Qwen model, create new instance for each call

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text}
    ]
    chat_response = qwen.chat(messages)  # Save LLM response to chat_response field
    new_state = state.copy()  # Create new state, add response entry, output new item
    new_state["chat_response"] = chat_response
    print("\n================================[SqlExecutionAgent]=================================\n")
    print("SqlExecutionAgent called chat response function:")
    print(chat_response)
    return new_state

