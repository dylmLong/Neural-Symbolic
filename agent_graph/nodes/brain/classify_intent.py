
from agent_graph.state_schema import AgentState, get_init_agent_state
from llm.qwen_wrapper import QwenWrapper


def classify_intent_node(state: AgentState) -> AgentState:
    user_input = state.get("user_text", "")
    qwen = QwenWrapper()
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classification assistant. You can only choose one of the two labels:\n"
                "1. chat: The user just wants to chat or ask a general question.\n"
                "2. reasoning: The user has uploaded an image and wants to analyze where it was taken, identify a location, or infer geographical information.\n"
                "Please reply strictly with only one label: chat or reasoning. Do not include any other content."
            )
        },
        {
            "role": "user",
            "content": f"User input:\n{user_input}\nPlease determine the user's intent type:"
        }
    ]
    intent = qwen.chat(messages).strip().lower()
    if intent not in {"chat", "reasoning"}:
        intent = "chat"
    new_state = state.copy()
    new_state["intent"] = intent
    return new_state

def route_intent_condition(state: AgentState):
    return state.get("intent")


if __name__ == '__main__':
    test_cases = [
        # 🔵 Chat-type inputs
        "Hello, who are you?",
        "How's the weather in Beijing recently?",
        "Can you tell me a joke?",
        "Please check the latest developments in AI.",
        "What do you think about Elon Musk?",

        # 🟢 Image analysis-type inputs
        "Please identify where this photo was taken.",
        "I’ve uploaded a photo—can you tell where it was taken?",
        "This picture seems to be near the sea. Can you confirm the exact location?",
        "Please analyze the geographical information behind this image.",
        "The building in this photo looks like it's somewhere in France—can you help confirm that?"
    ]
    for input_text in test_cases:
        print(f"\n=== Test Input ===\n{input_text}\n")
        # Construct state
        case_state: AgentState = get_init_agent_state(input_text, "xxx.png")

        # Call the classification node
        updated_state = classify_intent_node(case_state)

        # Output the intent recognition result
        print(f"👉 Recognized intent:{updated_state.get('intent')}")