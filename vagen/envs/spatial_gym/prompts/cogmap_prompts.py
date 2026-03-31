"""
Cognitive map prompt — global cognitive map only.
"""

GLOBAL_COGMAP_PROMPT = """\
## Global Cognitive Map (JSON)

Represent the scene as a JSON map.

### Schema
- position: [x, y] integers
- facing: direction the object's front face points — "north|south|east|west"

### Rules
- Include only observed objects.
- MUST include facing key for objects that have a facing direction.
- Grid: concise global map on an N×M grid.
- Frame: origin [0,0] is your initial position; your initial facing direction is north.
- Positions: derive each object's [x, y] from its observed grid coordinates during exploration.
- Content: include all observed objects and gates; include the agent.
- Facing: use "north|south|east|west" (cardinal direction only). Project diagonal headings to the nearest cardinal.

### Example
```json
{
    "agent": {"position": [2, 3], "facing": "east"},
    "chair": {"position": [2, 4], "facing": "north"},
    "sofa": {"position": [5, 1], "facing": "west"}
}
```
"""

LOCAL_PERCEPTION_PROMPT = """\
## Local Perception (JSON)

Describe all objects and doors you currently see in your field of view.

### Schema
- position: [x, y] integers relative to your current position
- facing: object's front face direction "+x|-x|+y|-y"

### Frame
- Origin [0, 0] is your current position.
- +y: your facing direction (forward)
- +x: right, -x: left, -y: backward

### Rules
- Include ALL visible objects and doors in your FOV.
- Use local axes for facing (+x/-x/+y/-y), NOT compass directions.

### Example
```json
{
    "origin": "agent",
    "objects": {
        "red chair": {"position": [1, 2], "facing": "-x"},
        "door A": {"position": [-1, 3], "facing": "+x"}
    }
}
```
"""
