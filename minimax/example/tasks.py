"""Small runnable business module used by the SDD demonstration."""

TASKS = [
    {"id": 1, "title": "Write specification", "status": "done"},
    {"id": 2, "title": "Implement filtering", "status": "open"},
    {"id": 3, "title": "Review tests", "status": "open"},
]


def list_tasks(status=None):
    """Return task copies in their original order."""
    return [dict(task) for task in TASKS]
