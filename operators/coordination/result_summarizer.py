"""
Result Summarizer Operator (Logical Operator)

Converts structured query results into natural language descriptions for better user experience.
"""
from typing import List, Dict, Any

from agent.shared.state import AgentState
from llm.qwen_wrapper import QwenWrapper


def format_results_for_llm(filter_results: List[Dict[str, Any]], detected_objects: List[Dict[str, Any]]) -> str:
    """
    Format filter results and detected objects into a structured text for LLM processing.
    
    :param filter_results: Filtered query results
    :param detected_objects: Detected objects from perception
    :return: Formatted text for LLM
    """
    # Format detected objects (only high confidence ones, exclude sunset)
    high_confidence_objects = [
        obj for obj in detected_objects 
        if obj.get('confidence', 0) > 0.7 and obj.get('label', '') != '夕阳'
    ]
    objects_text = "Key objects detected in the image:\n"
    object_labels = []
    for i, obj in enumerate(high_confidence_objects, 1):
        label = obj.get('label', 'Unknown')
        confidence = obj.get('confidence', 0)
        objects_text += f"{i}. {label} (confidence: {confidence:.0%})\n"
        object_labels.append(label)
    
    # Format location results - prioritize locations that match multiple detected objects
    locations_text = "\nGeographic location analysis results (sorted by relevance):\n"
    all_locations = []
    seen_pairs = set()
    
    for res in filter_results:
        result = res.get("result", [])
        sql = res.get("sql", "")
        for row in result:
            # Create a unique key to avoid duplicates
            loc_a = row.get('a_name', '')
            loc_b = row.get('b_name', '')
            pair_key = tuple(sorted([loc_a, loc_b]))
            
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                # Count how many detected objects match this location pair
                match_count = 0
                for label in object_labels:
                    if label in loc_a or label in loc_b:
                        match_count += 1
                
                location_info = {
                    "location_a_name": loc_a,
                    "location_a_address": row.get('a_address', ''),
                    "location_b_name": loc_b,
                    "location_b_address": row.get('b_address', ''),
                    "distance": row.get('distance', 0),
                    "match_count": match_count  # How many detected objects match
                }
                all_locations.append(location_info)
    
    # Sort by match_count (descending) then by distance (ascending)
    # Locations matching more detected objects are more relevant
    all_locations.sort(key=lambda x: (-x.get('match_count', 0), x.get('distance', float('inf'))))
    top_locations = all_locations[:3]  # Top 3 most relevant locations
    
    for i, loc in enumerate(top_locations, 1):
        match_info = f"matches {loc['match_count']} detected objects" if loc['match_count'] > 0 else "no direct object match"
        locations_text += (
            f"\nOption {i} ({match_info}, distance: {loc['distance']:.0f}m):\n"
            f"  - {loc['location_a_name']} ({loc['location_a_address']})\n"
            f"  - {loc['location_b_name']} ({loc['location_b_address']})\n"
        )
    
    return objects_text + locations_text


def summarize_result_operator(state: AgentState) -> AgentState:
    """
    LangGraph node function: Convert structured results into natural language description.
    
    This operator takes the filtered query results and generates a user-friendly natural language
    summary while preserving the original structured output.
    
    :param state: Agent state
    :return: Updated state (contains summary field)
    """
    filter_results = state.get("filter_results", [])
    detected_objects = state.get("objects", [])
    
    # If no results, return early
    if not filter_results or not detected_objects:
        new_state = state.copy()
        new_state["summary"] = "No location information could be determined from the image."
        return new_state
    
    # Format results for LLM
    formatted_data = format_results_for_llm(filter_results, detected_objects)
    
    # Use LLM to generate natural language description
    qwen = QwenWrapper()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes photo locations in a natural, friendly, and informative way. "
                "Based on the detected objects and geographic location analysis results, provide a clear and engaging summary "
                "answering where the photo was likely taken. "
                "\nGuidelines:\n"
                "- Prioritize locations that match multiple detected objects (higher match_count = more relevant)\n"
                "- Start with a direct answer about the most probable location (usually the one with highest match_count)\n"
                "- Mention the key landmarks or objects that helped identify the location\n"
                "- Include specific addresses, street names, or area names when available\n"
                "- If the top location matches multiple detected objects, emphasize this as strong evidence\n"
                "- Write in a conversational, user-friendly tone\n"
                "- Keep it concise: 2-4 sentences total\n"
                "- Be confident when match_count is high, but acknowledge uncertainty if locations are far apart or match_count is low\n"
                "- Focus on what makes the location identifiable (nearby landmarks, objects, etc.)"
            )
        },
        {
            "role": "user",
            "content": (
                f"Based on the following analysis, describe where this photo was likely taken:\n\n"
                f"{formatted_data}\n\n"
                f"Please provide a natural, friendly description of the most likely shooting location. "
                f"Start directly with your answer, mention key identifying features, and be specific about the location."
            )
        }
    ]
    
    summary = qwen.chat(messages)
    
    # Update state
    new_state = state.copy()
    new_state["summary"] = summary
    
    # Print the natural language summary
    print("\n================================[ResultSummarizer]=================================\n")
    print("ResultSummarizer generated natural language description:")
    print("\n" + "="*70)
    print(summary)
    print("="*70 + "\n")
    
    return new_state

