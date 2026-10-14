"""Run a command and an agent-written program inside a sandbox, with a boundary you can see."""

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from zett_agent.sandbox.base import Sandbox
from zett_agent.sandbox.errors import SandboxUnavailableError
from zett_agent.sandbox.factory import create_sandbox
from zett_agent.sandbox.passthrough import PassthroughSandbox
from zett_agent.sandbox.policy import NetworkMode, SandboxPolicy


def choose_sandbox(policy: SandboxPolicy) -> Sandbox:
    """Use this platform's sandbox, or run unisolated here because this example must work anywhere."""
    try:
        return create_sandbox(policy)
    except SandboxUnavailableError:
        # A real host would stop here, or fix the machine. The example only needs to run.
        return PassthroughSandbox(policy)


async def main() -> None:
    with TemporaryDirectory(prefix="zett-sandbox-") as directory:
        workspace = Path(directory).resolve()
        # Write this directory, read the machine, no network, protected files kept safe.
        policy = SandboxPolicy.workspace_write(workspace, network=NetworkMode.NONE)
        sandbox = choose_sandbox(policy)

        # A program the agent wrote runs through the same interface as any command.
        (workspace / "program.py").write_text(
            "from pathlib import Path\nPath('answer.txt').write_text('42')\nprint('ran')\n"
        )
        result = await sandbox.run([sys.executable, "program.py"], cwd=workspace, timeout=30)
        assert result.succeeded and result.stdout.strip() == "ran"
        assert (workspace / "answer.txt").read_text() == "42"

        # A shell string works too, and its output and exit status come back as data.
        listing = await sandbox.run("ls | sort", cwd=workspace)
        assert listing.stdout.split() == ["answer.txt", "program.py"]
        print("Sandbox ran the program and the command")


if __name__ == "__main__":
    asyncio.run(main())
