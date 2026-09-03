from src.app.core.graph.types import NodeError


def fault_metadata(error: NodeError) -> dict:
    """Build a serializable fault-tolerance metadata payload from a node error."""
    return {
        "failed_node": error.node,
        "error_type": type(error.error).__name__,
        "error_message": str(error.error),
    }
