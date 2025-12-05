"""
Unified Operator Pool (UOP)

According to the paper: The system requires an abstraction and registration mechanism that can define and integrate operators from all stages.
Allows users to register custom computation operators and treat them as building blocks.

Operator classification:
- Physical operators: Interact with external systems (YOLO, database execution)
- Logical operators: Data processing and decision-making (SQL generation, perception code generation, intent classification, routing, filtering)
"""
from typing import Dict, Callable, Any

# Operator registry: Records all available operators
OPERATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Perception operators (physical operators)
    "perception": {
        "yolo_detector": {
            "name": "YOLO Object Detection Operator",
            "type": "physical",
            "module": "operators.perception.yolo_detector",
            "function": "detect_objects",
            "description": "Detect objects in images using YOLOv8 model"
        }
    },
    # Execution operators (physical operators)
    "execution": {
        "sql_executor": {
            "name": "SQL Execution Operator",
            "type": "physical",
            "module": "operators.execution.sql_executor",
            "function": "execute_sql_query",
            "description": "Execute a single SQL query and return results"
        }
    },
    # Coding operators (logical operators)
    "coding": {
        "perception_code_generator": {
            "name": "Perception Code Generation Operator",
            "type": "logical",
            "module": "operators.coding.perception_code_generator",
            "function": "generate_perception_instruction",
            "description": "Generate execution instruction for YOLO perception call"
        },
        "sql_generator": {
            "name": "SQL Generation Operator",
            "type": "logical",
            "module": "operators.coding.sql_generator",
            "function": "generate_sql_queries",
            "description": "Encode perception results as SQL query statements"
        }
    },
    # Coordination operators (logical operators)
    "coordination": {
        "intent_classifier": {
            "name": "Intent Classification Operator",
            "type": "logical",
            "module": "operators.coordination.intent_classifier",
            "function": "classify_intent_operator",
            "description": "Classify intent type based on user input"
        },
        "intent_router": {
            "name": "Intent Routing Operator",
            "type": "logical",
            "module": "operators.coordination.intent_classifier",
            "function": "route_intent_condition",
            "description": "Determine next path based on intent"
        },
        "sql_router": {
            "name": "SQL Routing Operator",
            "type": "logical",
            "module": "operators.coordination.sql_router",
            "function": "sql_router_step",
            "description": "Select next SQL to execute"
        },
        "sql_router_condition": {
            "name": "SQL Routing Condition Operator",
            "type": "logical",
            "module": "operators.coordination.sql_router",
            "function": "route_sql_condition",
            "description": "Determine if there are more SQLs to execute"
        },
        "result_filter": {
            "name": "Result Filtering Operator",
            "type": "logical",
            "module": "operators.coordination.result_filter",
            "function": "filter_result_operator",
            "description": "Filter out SQLs with empty query results"
        },
        "chat_responder": {
            "name": "Chat Response Operator",
            "type": "logical",
            "module": "operators.coordination.chat_responder",
            "function": "chat_responder_operator",
            "description": "Use LLM to respond to user's chat requests"
        },
        "result_summarizer": {
            "name": "Result Summarizer Operator",
            "type": "logical",
            "module": "operators.coordination.result_summarizer",
            "function": "summarize_result_operator",
            "description": "Convert structured query results into natural language descriptions"
        }
    }
}


def get_operator(category: str, operator_name: str) -> Callable:
    """
    Get specified operator from operator pool.
    
    :param category: Operator category (perception, execution, coding, coordination)
    :param operator_name: Operator name
    :return: Operator function
    """
    if category not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator category: {category}")
    if operator_name not in OPERATOR_REGISTRY[category]:
        raise ValueError(f"Operator {operator_name} does not exist in category {category}")
    
    operator_info = OPERATOR_REGISTRY[category][operator_name]
    module_path = operator_info["module"]
    function_name = operator_info["function"]
    
    # Dynamically import operator
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def list_operators(category: str = None) -> Dict[str, Dict[str, Any]]:
    """
    List all available operators in operator pool.
    
    :param category: Optional, specify category
    :return: Operator information dictionary
    """
    if category:
        return OPERATOR_REGISTRY.get(category, {})
    return OPERATOR_REGISTRY
