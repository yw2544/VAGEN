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
    "agent": {"position": [0, 0], "facing": "north"},
    "chair": {"position": [2, 4], "facing": "north"},
    "sofa": {"position": [5, 1], "facing": "west"}
}
```
"""

LOCAL_PERCEPTION_PROMPT = """\
## Local Perception (JSON)

Describe all objects and doors you currently see in your field of view.

### Schema
- position: [x, y] integers relative to your current position (north = forward, east = right)
- facing: "north|northeast|east|southeast|south|southwest|west|northwest". Omit facing for objects without a meaningful front (e.g. vase, pillow).

### Rules
- Include ALL visible objects and doors in your FOV.
- North means the object faces the same direction as you (forward); east means it faces to your right; etc.

### Example
```json
{
    "origin": "agent",
    "objects": {
        "red chair": {"position": [1, 2], "facing": "northwest"},
        "vase": {"position": [0, 3]}
    }
}
```
"""
