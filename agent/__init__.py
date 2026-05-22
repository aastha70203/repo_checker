"""AI code review agent package."""

__all__ = ["CodeReviewAgent"]


def __getattr__(name: str):
    if name == "CodeReviewAgent":
        from agent.pipeline import CodeReviewAgent

        return CodeReviewAgent
    raise AttributeError(name)
