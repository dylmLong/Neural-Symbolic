"""
Result Summarizer Operator (Logical Operator)

Converts structured query results into natural language descriptions for better user experience.
"""
from typing import List, Dict, Any, Tuple

from agent.shared.state import AgentState
from llm.qwen_wrapper import QwenWrapper


def format_results_for_llm(filter_results: List[Dict[str, Any]], detected_objects: List[Dict[str, Any]]) -> Tuple[str, int]:
    """
    Format filter results and detected objects into a structured text for LLM processing.
    
    :param filter_results: Filtered query results
    :param detected_objects: Detected objects from perception
    :return: Tuple of (formatted text for LLM, number of available location options)
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
    num_locations = len(all_locations)
    top_locations = all_locations[:3]  # Top 3 most relevant locations
    
    for i, loc in enumerate(top_locations, 1):
        match_info = f"matches {loc['match_count']} detected objects" if loc['match_count'] > 0 else "no direct object match"
        locations_text += (
            f"\nOption {i} ({match_info}, distance: {loc['distance']:.0f}m):\n"
            f"  - {loc['location_a_name']} ({loc['location_a_address']})\n"
            f"  - {loc['location_b_name']} ({loc['location_b_address']})\n"
        )
    
    return objects_text + locations_text, num_locations


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
    
    # Format results for LLM and get the number of available location options
    formatted_data, num_locations = format_results_for_llm(filter_results, detected_objects)
    
    # Dynamically build prompt based on actual number of location options
    if num_locations == 0:
        new_state = state.copy()
        new_state["summary"] = "No location information could be determined from the image."
        return new_state
    elif num_locations == 1:
        # Only one location option - single paragraph
        format_instructions = (
            "Write a SINGLE paragraph describing the location. "
            "Start with 'This photo was likely taken...' and provide a detailed description "
            "including the specific address, nearby landmarks, and supporting evidence."
        )
        paragraph_guidance = (
            "Write exactly ONE paragraph. "
            "Start with 'This photo was likely taken...' and describe the location in detail."
        )
    elif num_locations == 2:
        # Two location options - two paragraphs
        format_instructions = (
            "Write TWO paragraphs, each separated by a blank line. "
            "Paragraph 1 (HIGHEST PRIORITY): Start with 'This photo was likely taken...' - describe Option 1. "
            "Paragraph 2 (SECOND PRIORITY): Start with 'Alternatively, it could also be...' or 'It could also be...' - describe Option 2."
        )
        paragraph_guidance = (
            "Write exactly TWO paragraphs, each separated by a blank line. "
            "Start the first paragraph with 'This photo was likely taken...', "
            "and the second with 'Alternatively, it could also be...' or 'It could also be...'."
        )
    else:
        # Three or more location options - three paragraphs (top 3)
        format_instructions = (
            "Write THREE paragraphs, each separated by a blank line. "
            "Paragraph 1 (HIGHEST PRIORITY): Start with 'This photo was likely taken...' - describe Option 1. "
            "Paragraph 2 (SECOND PRIORITY): Start with 'Alternatively, it could also be...' or 'It could also be...' - describe Option 2. "
            "Paragraph 3 (THIRD PRIORITY): Start with 'Another possibility is...' or 'Additionally, it might be...' - describe Option 3."
        )
        paragraph_guidance = (
            "Write exactly THREE paragraphs, each separated by a blank line. "
            "Start the first paragraph with 'This photo was likely taken...', "
            "the second with 'Alternatively, it could also be...' or 'It could also be...', "
            "and the third with 'Another possibility is...' or 'Additionally, it might be...'."
        )
    
    # Use LLM to generate natural language description
    qwen = QwenWrapper()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes photo locations in a natural, friendly, and informative way. "
                "Based on the detected objects and geographic location analysis results, provide a clear and engaging summary "
                "answering where the photo was likely taken. "
                f"\nIMPORTANT FORMATTING REQUIREMENTS:\n"
                f"{format_instructions}"
                "\n\nContent Guidelines:\n"
                "- Prioritize locations that match multiple detected objects (higher match_count = more relevant)\n"
                "- Mention the key landmarks or objects that helped identify each location\n"
                "- Include specific addresses, street names, or area names when available\n"
                "- Mention the distance between landmarks when relevant\n"
                "- Write in a conversational, user-friendly tone\n"
                "- Each paragraph should be 2-3 sentences\n"
                "- Be confident for the first paragraph (highest match_count), but acknowledge lower confidence for subsequent options\n"
                "- Focus on what makes each location identifiable (nearby landmarks, objects, etc.)"
            )
        },
        {
            "role": "user",
            "content": (
                f"Based on the following analysis, describe where this photo was likely taken:\n\n"
                f"{formatted_data}\n\n"
                f"{paragraph_guidance}"
            )
        }
    ]
    
    summary = qwen.chat(messages)
    
    # Clean up the summary: preserve paragraph structure while removing duplicates
    if summary:
        import re
        # Remove leading/trailing whitespace
        summary = summary.strip()
        # Preserve paragraph breaks (double newlines) but normalize single newlines within paragraphs
        # First, normalize multiple spaces to single space within paragraphs
        summary = re.sub(r'[ \t]+', ' ', summary)
        # Preserve paragraph breaks (double newlines or more)
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        # Normalize single newlines within paragraphs to spaces (but keep double newlines for paragraphs)
        lines = summary.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:  # Non-empty line
                cleaned_lines.append(stripped)
            elif cleaned_lines and cleaned_lines[-1]:  # Empty line after content (paragraph break)
                cleaned_lines.append('')
        
        # Reconstruct with proper paragraph breaks
        summary = '\n\n'.join([line for line in cleaned_lines if line])
        
        # Remove obvious repetition within each paragraph
        paragraphs = summary.split('\n\n')
        cleaned_paragraphs = []
        for para in paragraphs:
            if para.strip():
                # Simple deduplication: remove duplicate sentences within the same paragraph
                sentences = re.split(r'([.!?]\s+)', para)
                cleaned_sentences = []
                seen_in_para = set()
                
                for i in range(0, len(sentences), 2):
                    if i < len(sentences):
                        text = sentences[i].strip()
                        punct = sentences[i+1] if i+1 < len(sentences) else ''
                        
                        if text and text.lower() not in seen_in_para:
                            # Check if this sentence is very similar to a previous one
                            is_duplicate = False
                            for seen in seen_in_para:
                                if text.lower() in seen.lower() or seen.lower() in text.lower():
                                    if abs(len(text) - len(seen)) < max(len(text), len(seen)) * 0.3:
                                        is_duplicate = True
                                        break
                            
                            if not is_duplicate:
                                cleaned_sentences.append(text + punct)
                                seen_in_para.add(text.lower())
                
                if cleaned_sentences:
                    cleaned_paragraphs.append(' '.join(cleaned_sentences))
        
        summary = '\n\n'.join(cleaned_paragraphs)
        
        # Ensure it ends with punctuation
        if summary and not summary[-1] in '.!?':
            summary += '.'
    
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

