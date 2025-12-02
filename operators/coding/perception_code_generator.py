"""
Perception Code Generation Operator (Logical Operator)

Generates execution instruction for YOLO perception call based on Coding Agent's requirements.
"""
from typing import Dict, Any

from agent.shared.state import AgentState, ExecutionInstruction


def generate_perception_instruction(state: AgentState) -> ExecutionInstruction:
    """
    Generate perception call instruction: Coding Agent generates execution instruction for YOLO detection.
    
    :param state: Agent state, contains image_path
    :return: Execution instruction, contains operator name and parameters
    """
    image_path = state.get("image_path")
    if not image_path:
        raise ValueError("image_path not provided, cannot generate perception instruction")
    
    # Generate execution instruction: call YOLO detection operator
    instruction: ExecutionInstruction = {
        "operator": "yolo_detector",  # Operator name
        "params": {
            "image_path": image_path  # Operator parameters
        }
    }
    return instruction

