"""
Result Filtering Operator (Logical Operator)

Filters out SQLs with empty query results, combines SQLs with results and data for output.
"""
from typing import List, Dict, Any

from agent.shared.state import AgentState


def filter_results(sql_statements: List[str], query_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Filter out SQLs with empty query results, combine SQLs with results and data for output.
    
    :param sql_statements: List of SQL statements
    :param query_results: Query results for each SQL statement
    :return: SQL + Result pairs containing actual results
    """
    valid_results = []
    for sql, result in zip(sql_statements, query_results):
        if isinstance(result, list) and len(result) > 0:
            valid_results.append({
                "sql": sql,
                "result": result
            })
    return valid_results


def filter_result_operator(state: AgentState) -> AgentState:
    """
    LangGraph node function: Filter out SQLs with empty query results, combine SQLs with results and data for output.
    
    :param state: Agent state
    :return: Updated state (contains filter_results)
    """
    print("\n================================[SqlExecutionAgent]=================================\n")
    print("SqlExecutionAgent called result filtering function:")
    print("Summarizing valid results based on executed SQLs and query results:")
    results = filter_results(
        state["sql_statements"],
        state["query_results"]
    )

    for i, res in enumerate(results, 1):
        print(f"\nValid SQL: {res.get('sql')}")
        print(f"Execution result:")
        result = res.get("result")
        print("[")
        for row in result:
            print(f"\t {row}")
        print("]")
    new_state = state.copy()
    new_state["filter_results"] = results

    print("\nBased on confidence, this image was most likely taken at one of the following locations:")
    for i, res in enumerate(results, 1):
        result = res.get("result")
        for idx, row in enumerate(result, 1):
            print(f"[{idx}]: {row}")
        if i > 0:
            break

    return new_state

