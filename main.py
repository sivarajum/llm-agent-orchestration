"""Entry point: launch the Agent Orchestration API and/or UI."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def main() -> None:
    """Parse CLI args and launch the requested mode."""
    # Set up logging before anything else
    from src.logging_config import setup_logging

    setup_logging()

    from src.settings import API_HOST, API_PORT, UI_PORT

    mode = sys.argv[1] if len(sys.argv) > 1 else "api"

    if mode == "api":
        import uvicorn

        uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=True)

    elif mode == "ui":
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", "src/ui.py",
                "--server.port", str(UI_PORT), "--server.address", "0.0.0.0",
            ],
            cwd=PROJECT_ROOT,
        )

    elif mode == "all":
        api_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "src.api:app",
                "--host", API_HOST, "--port", str(API_PORT),
            ],
            cwd=PROJECT_ROOT,
        )
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "streamlit", "run", "src/ui.py",
                    "--server.port", str(UI_PORT), "--server.address", "0.0.0.0",
                ],
                cwd=PROJECT_ROOT,
            )
        finally:
            api_proc.terminate()

    elif mode == "run":
        # Run pipeline directly from CLI
        topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "AI agents"
        from src.orchestrator import run_pipeline

        result = run_pipeline(topic)
        print(f"\n{'='*60}")
        print(f"  Topic: {result['topic']}")
        print(f"  Status: {result['status']}")
        print(f"  Iterations: {result['iteration']}")
        print(f"{'='*60}")
        print(f"\n--- Research ---\n{result['research']}")
        print(f"\n--- Draft ---\n{result['draft']}")
        print(f"\n--- Review ---\n{result['review_feedback']}")

    else:
        print("Usage: python main.py [api|ui|all|run <topic>]")
        print("  api            - Start the FastAPI server (default)")
        print("  ui             - Start the Streamlit dashboard")
        print("  all            - Start both API and UI")
        print("  run <topic>    - Run the pipeline directly in the terminal")
        sys.exit(1)


if __name__ == "__main__":
    main()
