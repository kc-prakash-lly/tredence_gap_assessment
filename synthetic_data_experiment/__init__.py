"""synthetic_data_experiment — gap assessment ADK package."""
# root_agent is exposed here so `adk web` can discover it.
# agent.py is written in Step 2; this import is a no-op stub until then.
try:
    from .agent import root_agent  # noqa: F401
except ImportError:
    pass
