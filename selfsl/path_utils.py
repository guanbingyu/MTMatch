from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_task_dir(task_type: str, task: str) -> str:
    """Resolve a task directory for packaged or workspace-style layouts."""
    project_root = _project_root()
    candidates = []

    if task_type == "em":
        candidates.append(project_root / "data" / "em" / task)
        candidates.append(project_root / "shared_data" / "ER_EM" / task)
        candidates.append(project_root.parent / "shared_data" / "ER_EM" / task)
    else:
        candidates.append(project_root / "data" / task_type / task)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


def resolve_task_file(task_type: str, task: str, filename: str) -> str:
    return str(Path(resolve_task_dir(task_type, task)) / filename)
