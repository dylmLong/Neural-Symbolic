"""
SQL Routing Operator (Logical Operator)

Responsible for SQL query scheduling, selecting the next SQL to execute, and determining if there are more SQLs to execute.
"""
from agent.shared.state import AgentState


def sql_router_step(state: AgentState) -> AgentState:
    """
    LangGraph node function: SQL routing, select next SQL to execute
    
    :param state: Agent state
    :return: Updated state (contains current_sql and current_index)
    """
    executed = set(state.get("executed_sqls") or [])  # Already executed SQLs
    sql_list = state.get("sql_statements") or []  # Total SQL statements
    remaining = [sql for sql in sql_list if sql not in executed]  # Remaining SQL statements = total SQL minus already executed

    if not remaining:  # No remaining, end directly
        return state

    current_sql = remaining[0]  # Get currently executing SQL statement

    new_state = state.copy()  # Set new state, record current executing SQL statement content and index value in new state, return new state
    new_state["current_sql"] = current_sql
    new_state["current_index"] = len(executed) + 1
    return new_state


def route_sql_condition(state: AgentState) -> str:
    """
    Routing condition function: Determine if there are more SQLs to execute
    
    :param state: Agent state
    :return: "continue" or "done"
    """
    remaining = set(state.get("sql_statements") or []) - set(state.get("executed_sqls") or [])
    return "continue" if remaining else "done"

