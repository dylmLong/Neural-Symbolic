from agent_graph.state_schema import AgentState, get_init_agent_state
from llm.qwen_wrapper import QwenWrapper


def llm_chat_node(state: AgentState) -> AgentState:
    user_text = state.get("user_text", "")
    qwen = QwenWrapper()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text}
    ]
    chat_response = qwen.chat(messages)
    new_state = state.copy()
    new_state["chat_response"] = chat_response
    print("\n================================[Brain Message]=================================\n")
    print(chat_response)
    return new_state


if __name__ == '__main__':
    test_cases = [
        "Hello, how's the weather today?"
        "Who are you?"
        "Can you tell me a joke?"
        "Can you recommend some good books?"
        "What is quantum entanglement?"
    ]

    for input_text in test_cases:
        case_state: AgentState = get_init_agent_state(input_text, "")
        updated_state = llm_chat_node(case_state)
        # Output Model Response
        print(f"question:{input_text}")
        print("💬 Model Response:")
        print(updated_state.get("chat_response", "no reply"))
