from modes.base import DetectionMode, RuntimeContext

MODE_NAMES = ("vehicle", "person", "idle")


def build_mode(name: str, ctx: RuntimeContext) -> DetectionMode:
    """Create a mode. Imports are lazy on purpose: a mode's models and libraries are only loaded when
    that mode is actually selected, and a missing dependency of one mode can't break the others."""
    if name == "vehicle":
        from modes.vehicle.mode import VehicleMode
        return VehicleMode(ctx)
    if name == "person":
        from modes.person.mode import PersonMode
        return PersonMode(ctx)
    if name == "idle":
        from modes.idle import IdleMode
        return IdleMode(ctx)
    raise ValueError(f"unknown detection mode {name!r} (known: {', '.join(MODE_NAMES)})")
