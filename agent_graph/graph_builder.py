# from gradio.monitoring_dashboard import process_time
from langgraph.graph import StateGraph
from agent_graph.state_schema import AgentState, get_init_agent_state
from agent_graph.nodes.perception.detect import detect_node
from agent_graph.nodes.brain.classify_intent import classify_intent_node, route_intent_condition
from agent_graph.nodes.brain.llm_chat import llm_chat_node
from agent_graph.nodes.brain.generate_sql import generate_sql_node
from agent_graph.nodes.brain.sql_router import sql_router_node, route_sql_condition
from agent_graph.nodes.brain.filter_result import filter_result_node
from agent_graph.nodes.brain.generate_answer import generate_answer_node
from agent_graph.nodes.action.execute_sql import execute_sql_node


def build_agent_graph():
    # Construct state graph
    graph = StateGraph(AgentState)

    # Intent recognition node
    graph.add_node("classify_intent", classify_intent_node)

    # chatting node
    graph.add_node("llm_chat", llm_chat_node)

    # detection node
    graph.add_node("detect", detect_node)

    # SQL generation node
    graph.add_node("generate_sql", generate_sql_node)

    # SQL routing node
    graph.add_node("sql_router", sql_router_node)

    # SQL execution node
    graph.add_node("execute_sql", execute_sql_node)

    # result filtering node
    graph.add_node("filter_result", filter_result_node)

    # Set up a question-answering node based on the results
    # graph.add_node("generate_answer", generate_answer_node)

    # Set the entry point
    graph.set_entry_point("classify_intent")

    # Intent router
    graph.add_conditional_edges("classify_intent", route_intent_condition, {
        "chat": "llm_chat",
        "reasoning": "detect",
    })

    graph.add_edge("detect", "generate_sql")
    graph.add_edge("generate_sql", "sql_router")
    graph.add_edge("execute_sql", "sql_router")

    # Conditional loop
    graph.add_conditional_edges("sql_router", route_sql_condition, {
        "continue": "execute_sql",
        "done": "filter_result"
    })
    # graph.add_edge("filter_result", "generate_answer")

    # Set the finish point
    # graph.set_finish_point("generate_answer")
    graph.set_finish_point("filter_result")
    graph.set_finish_point("llm_chat")

    return graph.compile()

def test_case1():
    # Construct the initial state
    image_path = "../data/test1.jpg"
    user_text = "Where was this photo taken?"
    state: AgentState = get_init_agent_state(user_text, image_path)


    print("================================[Human Message]=================================")
    print(state.get("user_text"))
    print(state.get("image_path"))
    # Build and run LangGraph()
    graph = build_agent_graph()
    # graph.get_graph().print_ascii()
    final_state = graph.invoke(state, {"recursion_limit": 100})

def test_case2():
    # Construct the initial state
    user_text2 = "Who are you?"
    state: AgentState = get_init_agent_state(user_text2, "")
    print("\n================================[Human Message]=================================\n")
    print(state.get("user_text"))
    # Build and run LangGraph()
    graph = build_agent_graph()
    # graph.get_graph().print_ascii()
    final_state = graph.invoke(state, {"recursion_limit": 100})

if __name__ == '__main__':
    test_case1()
