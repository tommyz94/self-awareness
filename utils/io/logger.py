class Logger:
    """
    A simple file-based logger for writing structured log entries.

    Usage:
        logger = Logger()           # uses config.OUTPUT_FILE by default
        logger.reset()              # clears existing log file
        logger.log(
            role="Assistant",
            iteration=1,
            reasoning="Some reasoning text",
            action="Some action",
            summary="A brief summary"
        )
    """
    def __init__(self, output_file: str = None):
        self.output_file = output_file

    def reset(self) -> None:
        """Initialize or clear the log file."""
        from pathlib import Path

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            f.write("")

    def log(
        self,
        role: str,
        iteration: int,
        reasoning: str,
        action: str,
        summary: str,
        thinking: str = None
    ) -> None:
        """
        Append a structured log entry to the log file.

        Each entry contains role, iteration count, thinking (if available), reasoning,
        action taken, and a summary, separated by dividers.
        """
        entry = f"Role: {role}; Iteration {iteration}\n"
        if thinking:
            entry += f"Thinking: {thinking}\n"
        entry += (
            f"Reasoning: {reasoning}\n"
            f"Action: {action}\n"
            f"Summary: {summary}\n"
            "---------------------\n"
        )
        from pathlib import Path

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "a") as log_file:
            log_file.write(entry)

        # Also output iteration and action to the terminal
        print(f"Iteration {iteration}: {action}")
