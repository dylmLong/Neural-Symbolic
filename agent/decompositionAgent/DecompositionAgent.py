"""
Decomposition Agent

According to the paper: The decomposition agent first automatically explores possible combinations from the unified operator pool (UOP),
and generates a series of unique, well-defined subtasks that can complete the user's task.

Responsibilities:
1. Select appropriate operators from the operator pool (UOP)
2. Classify user intent and determine execution flow direction (intent classification and routing)
3. Compose operators into an execution plan (LangGraph DAG)
4. Serve as the entry point of the entire system
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import StateGraph

from agent.codingAgent.CodingAgent import CodingAgent
from agent.coordinationAgent.CoordinationAgent import CoordinationAgent
from agent.executionAgent.ExecutionAgent import ExecutionAgent
from agent.shared.state import AgentState, get_init_agent_state
from operators import OPERATOR_REGISTRY

# Select intent classification operator from operator pool (core responsibility of Decomposition Agent)
from operators.coordination.intent_classifier import classify_intent_operator, route_intent_condition
# Select result summarizer operator from operator pool
from operators.coordination.result_summarizer import summarize_result_operator


class DecompositionAgent:
    """
    Decomposition Agent: Selects operators from the unified operator pool (UOP) and composes execution plans.
    
    According to the paper, the decomposition agent's responsibilities are:
    - Automatically explore possible combinations from the unified operator pool (UOP)
    - Generate a series of unique, well-defined subtasks
    - Build execution plans (LangGraph DAG)
    
    Core responsibilities of Decomposition Agent:
    - Classify user intent (intent classification)
    - Determine execution flow direction based on intent (intent routing)
    - Select operators from operator pool and compose execution plan
    
    New execution flow (interface separation):
    - Reasoning path: intent classification -> generate perception code -> execute perception -> generate SQL -> execute SQL
    
    Architecture design:
    - Instantiate two Coding Agent objects: one for generating Python/YOLO call code, one for generating SQL
    - Instantiate two Execution Agent objects: one for executing perception (YOLO), one for executing SQL, routing, filtering, chat
    - This design facilitates future parallel processing
    """
    def __init__(self):
        # Instantiate Coding Agents (two objects for future parallel processing)
        self.perception_coding_agent = CodingAgent()  # For generating Python/YOLO call code
        self.sql_coding_agent = CodingAgent()          # For generating SQL query code
        
        # Instantiate Execution Agents (two objects for future parallel processing)
        self.perception_execution_agent = ExecutionAgent()  # For executing perception (YOLO)
        self.sql_execution_agent = ExecutionAgent()          # For executing SQL, routing, filtering, chat
        
        # Instantiate Coordination Agent
        self.coordination_agent = CoordinationAgent()  # Select from operator pool: coordination operators (feedback, error handling, rollback, retry)
        
        # Operator registry (UOP - Unified Operator Pool)
        self.operator_registry = OPERATOR_REGISTRY

    def classify_intent(self, state: AgentState) -> AgentState:
        """
        Core responsibility of Decomposition Agent: Classify user intent
        
        Select from operator pool: operators.coordination.intent_classifier.classify_intent_operator
        This is the first step of task decomposition, determining the direction of the entire execution flow.
        """
        print("\n================================[DecompositionAgent]=================================\n")
        print("DecompositionAgent called intent classification function:")
        result_state = classify_intent_operator(state)
        intent = result_state.get("intent", "unknown")
        print(f"Intent classification result: {intent}")
        return result_state

    def route_intent(self, state: AgentState) -> str:
        """
        Core responsibility of Decomposition Agent: Determine execution flow direction based on intent
        
        Select from operator pool: operators.coordination.intent_classifier.route_intent_condition
        Based on intent classification result, decide whether to take chat path or reasoning path.
        """
        route = route_intent_condition(state)
        print(f"DecompositionAgent called intent routing function, routing to: {route} path")
        return route

    def _select_operators(self):
        """
        Select operators from operator pool to build the operator combination required for execution plan.
        
        According to the paper, this is the core function of the decomposition agent: selecting operators from the unified operator pool (UOP).
        Current implementation is hardcoded selection, can be extended to dynamic selection in the future.
        """
        # Select operators from operator pool (hardcoded approach, meets current requirements)
        selected_operators = {
            # Decomposition operators: intent classification (called directly by Decomposition Agent)
            "intent_classifier": self.classify_intent,
            "intent_router": self.route_intent,
            # Execution operators: chat response (called through SQL Execution Agent)
            "chat_responder": self.sql_execution_agent.chat,
            # Coding operators: generate perception code (called through Perception Coding Agent)
            "generate_perception_code": self.perception_coding_agent.generate_perception_code,
            # Execution operators: execute perception (called through Perception Execution Agent, executes instructions generated by Coding Agent)
            "execute_perception": self.perception_execution_agent.execute_perception,
            # Coding operators: SQL generation (called through SQL Coding Agent)
            "sql_generator": self.sql_coding_agent.generate_sql,
            # Execution operators: SQL routing (called through SQL Execution Agent)
            "sql_router": self.sql_execution_agent.sql_router_step,
            "sql_router_condition": self.sql_execution_agent.route_sql,
            # Execution operators: SQL execution (called through SQL Execution Agent)
            "sql_executor": self.sql_execution_agent.execute_sql,
            # Execution operators: result filtering (called through SQL Execution Agent)
            "result_filter": self.sql_execution_agent.filter_result,
            # Coordination operators: result summarization (standalone operator, not tied to specific agent)
            "result_summarizer": summarize_result_operator,
        }
        return selected_operators

    def build_graph(self):
        """
        Select operators from operator pool and compose execution plan (LangGraph DAG).
        
        According to the paper, this is the core function of the decomposition agent:
        - Select operators from the unified operator pool (UOP)
        - Compose operators into a data flow graph (DAG)
        
        New execution flow (interface separation):
        - Reasoning path: intent classification -> generate perception code -> execute perception -> generate SQL -> execute SQL
        """
        # Select operators from operator pool
        operators = self._select_operators()
        
        # Build state graph
        graph = StateGraph(AgentState)

        # Add selected operators to execution plan
        # Intent classification operator (called directly by Decomposition Agent)
        graph.add_node("classify_intent", operators["intent_classifier"])

        # Chat operator (called through SQL Execution Agent)
        graph.add_node("llm_chat", operators["chat_responder"])

        # Generate perception code operator (called through Perception Coding Agent)
        graph.add_node("generate_perception_code", operators["generate_perception_code"])

        # Execute perception operator (called through Perception Execution Agent, executes instructions generated by Coding Agent)
        graph.add_node("execute_perception", operators["execute_perception"])

        # SQL generation operator (called through SQL Coding Agent)
        graph.add_node("generate_sql", operators["sql_generator"])

        # SQL routing operator (called through SQL Execution Agent)
        graph.add_node("sql_router", operators["sql_router"])

        # SQL execution operator (called through SQL Execution Agent)
        graph.add_node("execute_sql", operators["sql_executor"])

        # Result filtering operator (called through SQL Execution Agent)
        graph.add_node("filter_result", operators["result_filter"])

        # Result summarization operator (standalone, converts structured results to natural language)
        graph.add_node("summarize_result", operators["result_summarizer"])

        # Set entry point
        graph.set_entry_point("classify_intent")

        # Intent routing: select path based on intent classification result (core responsibility of Decomposition Agent)
        graph.add_conditional_edges("classify_intent", operators["intent_router"], {
            "chat": "llm_chat",
            "reasoning": "generate_perception_code",  # Reasoning path: first generate perception code
        })

        # Reasoning path: generate perception code -> execute perception -> generate SQL -> route -> execute (loop) -> filter
        graph.add_edge("generate_perception_code", "execute_perception")  # Perception Coding Agent generates code, Perception Execution Agent executes
        graph.add_edge("execute_perception", "generate_sql")  # After executing perception, generate SQL
        graph.add_edge("generate_sql", "sql_router")
        graph.add_edge("execute_sql", "sql_router")
        
        # SQL execution loop: routing determines whether to continue execution
        graph.add_conditional_edges("sql_router", operators["sql_router_condition"], {
            "continue": "execute_sql",
            "done": "filter_result"
        })

        # After filtering results, generate natural language summary
        graph.add_edge("filter_result", "summarize_result")

        # Set finish points
        graph.set_finish_point("summarize_result")
        graph.set_finish_point("llm_chat")

        return graph.compile()

    def run(self, user_text: str, image_path: str):
        """
        Run Decomposition Agent: Select operators from operator pool, build and execute execution plan.
        
        This is the system entry point, corresponding to the original graph_builder.
        """
        # Construct initial state
        state: AgentState = get_init_agent_state(user_text, image_path)

        print("================================[DecompositionAgent]=================================")
        print("DecompositionAgent received user input:")
        print(f"User text: {state.get('user_text')}")
        if state.get("image_path"):
            print(f"Image path: {state.get('image_path')}")

        # Select operators from operator pool, build execution plan
        graph = self.build_graph()
        
        # Execute plan
        final_state = graph.invoke(state, {"recursion_limit": 100})
        return final_state


# System entry point: DecompositionAgent is the graph_builder
if __name__ == '__main__':
    # Test case 1: Reasoning task
    decomposition_agent = DecompositionAgent()
    decomposition_agent.run(
        user_text="Where was this photo taken?",
        image_path="data/test3.jpg"
    )
    
    # Test case 2: Chat task
    # decomposition_agent.run(
    #     user_text="Who are you?",
    #     image_path=""
    # )
