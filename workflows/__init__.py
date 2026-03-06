from .examination import ExaminationWorkflow
from .custom import CustomWorkflow
from .roleplay import RoleplayWorkflow
from .fill_gaps import FillGapsWorkflow
from .analogous import AnalogousWorkflow
from .reflection import ReflectionWorkflow
from .agent_builder import AgentBuilderWorkflow
from .base import BaseWorkflow, WorkflowContext
from .memory_agent import MemoryAgentWorkflow

__all__ = [
    "ExaminationWorkflow",
    "CustomWorkflow",
    "RoleplayWorkflow",
    "FillGapsWorkflow",
    "AnalogousWorkflow",
    "ReflectionWorkflow",
    "AgentBuilderWorkflow",
    "BaseWorkflow",
    "WorkflowContext",
    "MemoryAgentWorkflow",
]

WORKFLOW_REGISTRY = {
    12: ExaminationWorkflow,
    25: CustomWorkflow,
    26: FillGapsWorkflow,
    27: RoleplayWorkflow,
    28: ReflectionWorkflow,
    29: AnalogousWorkflow,
    33: MemoryAgentWorkflow,
}

def get_workflow_class(template_id: int):
    """Get workflow class by template_id."""
    return WORKFLOW_REGISTRY.get(template_id)

def get_agent_builder_workflow():
    """Get Agent Builder workflow class for OpenAI workflows."""
    return AgentBuilderWorkflow