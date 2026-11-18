from agent_impl import Agent as ClaudeAgent


class TwoRoundContext:
    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        return getattr(self.context, name)

    def status(self):
        state = self.context.status()
        remaining = min(state["evaluations_remaining"], max(0, 2 - len(state["history"])))
        return {
            **state,
            "evaluations_remaining": remaining,
            "evaluation_dispatch_allowed": remaining > 0 and state["evaluation_dispatch_allowed"],
        }


class Agent:
    def run(self, context):
        return ClaudeAgent().run(TwoRoundContext(context))
