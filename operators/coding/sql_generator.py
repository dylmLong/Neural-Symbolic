"""
SQL Generation Operator (Logical Operator)

Encodes perception results (object list) as SQL query statements.
Generates SQL queries for geographic locations based on detected object labels and spatial relationships.
"""
import re
from typing import List, Optional

from agent.shared.state import ObjectBox


def compare_relative_longitude(a: ObjectBox, b: ObjectBox, sunset: ObjectBox) -> str:
    """
    Determine whether a is west or east of b relative to sunset
    
    :param a: Object A
    :param b: Object B
    :param sunset: Sunset object (as directional reference)
    :return: "WEST" or "EAST"
    """
    def cx(box: ObjectBox) -> float:  # Define object center coordinates using bounding box coordinates
        return (box['bbox'][0] + box['bbox'][2]) / 2.0
    sunset_cx = cx(sunset)
    a_cx = cx(a)
    b_cx = cx(b)
    return "WEST" if abs(a_cx - sunset_cx) < abs(b_cx - sunset_cx) else "EAST"
    # abs(a's coordinate - sunset coordinate) < abs(b's coordinate - sunset coordinate) means a is horizontally closer to sunset in the image
    # Sunset is always in the west direction, being horizontally closer to sunset in the image means its position is more to the west in the real world


# detect.py calls YOLOv8 model to identify objects, here we use object bounding box coordinates
# Object coordinates in absolute direction (sunset is absolutely in the west) reflect the spatial relationship of objects
# Set SQL query statement condition part based on different object spatial relationships

def generate_spatial_sql(a: ObjectBox, b: ObjectBox, sunset: Optional[ObjectBox]) -> str:
    """
    Generate SQL query for spatial relationship between two objects
    
    :param a: Object A
    :param b: Object B
    :param sunset: Sunset object (optional, used for direction judgment)
    :return: SQL query statement
    """
    direction_condition = ""
    if sunset:
        direction = compare_relative_longitude(a, b, sunset)
        if direction == "WEST":  # Set SQL query statement condition based on west or east direction
            direction_condition = " AND ST_X(a.location) <= ST_X(b.location) "
        else:
            direction_condition = " AND ST_X(b.location) <= ST_X(a.location) "

    sql = f"""
        WITH geo_a AS (
            SELECT * FROM geo_table WHERE name LIKE '%{a['label']}%'
        ), geo_b AS (
            SELECT * FROM geo_table WHERE name LIKE '%{b['label']}%'
        )
        SELECT
            a.name AS a_name,
            a.address AS a_address,
            b.name AS b_name,
            b.address AS b_address,
            ROUND((st_distance(a.location, b.location) / 0.0111) * 1000) AS distance
        FROM geo_a AS a
        JOIN geo_b AS b ON 1=1
        WHERE 1=1 AND distance > 1 AND distance < 100
        {direction_condition}
        ORDER BY distance
        LIMIT 3;
    """
    return re.sub(r'\s+', ' ', sql.strip())
    # Spatial query, uses DuckDB's spatial function st_distance to calculate distance
    # Distance filtering, only returns results within 1-100 meters between a and b
    # Result sorting, sorted by distance in ascending order, take top 3


def generate_sql_queries(objects: List[ObjectBox]) -> List[str]:
    """
    Generate SQL query statements for all object combinations
    Generate SQL query statements based on detected object labels
    
    :param objects: List of detected objects
    :return: List of SQL query statements
    """
    sql_list = []
    sunset_obj = next((obj for obj in objects if obj["label"] == "夕阳"), None)
    # Enumerate all possible object pair (a, b) combinations: a != b
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj_a = objects[i]
            obj_b = objects[j]

            if obj_a["label"] == "夕阳" or obj_b["label"] == "夕阳" or obj_a["label"] == obj_b["label"]:
                continue  # Sunset is only used as reference, not included in query, same labels are also skipped
            sql = generate_spatial_sql(obj_a, obj_b, sunset_obj)
            sql_list.append(sql)
    return sql_list

# Sunset is not treated as an object, here we have obtained SQL queries for spatial relationships between all object pairs

