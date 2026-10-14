"""Run an agent-written program in a sandbox that cleans up after itself."""

import sys

from zett_agent.sandbox.errors import SandboxUnavailableError
from zett_agent.sandbox.factory import create_sandbox
from zett_agent.sandbox.session import SandboxSession

PROGRAM = "from pathlib import Path\nPath('answer.txt').write_text('42')\nprint('ran')\n"


def pick_backend() -> str:
    """Use this platform's sandbox, or run unisolated because this example must work anywhere."""
    try:
        create_sandbox()
    except SandboxUnavailableError:
        # A real host would stop here, or fix the machine. The example only needs to run.
        return "passthrough"
    return "auto"


def main() -> None:
    # Say what the program may use; the workspace is a private directory that is deleted on exit.
    with SandboxSession(memory="512MB", cpu_seconds=30, max_file_size="10MB", backend=pick_backend()) as sb:
        workspace = sb.workspace
        (workspace / "program.py").write_text(PROGRAM)

        # Like subprocess.run: a list runs the program directly, check=True raises on failure.
        result = sb.run([sys.executable, "program.py"], check=True)
        assert result.stdout.strip() == "ran"
        assert (workspace / "answer.txt").read_text() == "42"

        # A string runs through the shell, so pipes work.
        assert sb.run("ls | sort").stdout.split() == ["answer.txt", "program.py"]

        # A command that writes past its file-size limit is stopped by the kernel.
        huge = sb.run("head -c 50000000 /dev/zero > big.bin")
        assert not huge.succeeded
        assert (workspace / "big.bin").stat().st_size <= 10 * 1024**2

    # Leaving the block removed the workspace and everything in it.
    assert not workspace.exists()
    print("Sandbox ran the program and the command")


if __name__ == "__main__":
    main()
