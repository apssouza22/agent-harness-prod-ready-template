"""Script to interact with the Deep Research API: login, create session, and submit research queries.

Usage:
    python src/cli/deep_research_client.py --email user@example.com --password yourpassword
    python src/cli/deep_research_client.py --email user@example.com --password yourpassword --stream
    python src/cli/deep_research_client.py --email user@example.com --password yourpassword --register
"""

import argparse
import json
import sys

import httpx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Deep Research API client for submitting research queries.")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--email", required=True, help="User email for login")
    parser.add_argument("--password", required=True, help="User password for login")
    parser.add_argument("--register", action="store_true", help="Register a new user before login")
    parser.add_argument("--stream", action="store_true", help="Use streaming endpoint for research responses")
    return parser.parse_args()


def register_user(client: httpx.Client, api_v1: str, email: str, password: str) -> dict:
    """Register a new user.

    Args:
        client: The HTTP client.
        api_v1: The API v1 base URL.
        email: User email.
        password: User password.

    Returns:
        dict: The registration response.
    """
    console.print("\n[bold cyan]1. Registering user...[/bold cyan]")
    response = client.post(
        f"{api_v1}/auth/register",
        json={"email": email, "password": password},
    )

    if response.status_code != 200:
        console.print(f"[bold red]Registration failed:[/bold red] {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json()
    console.print(Panel(f"User registered: [green]{data['email']}[/green] (id: {data['id']})", title="Registration"))
    return data


def login(client: httpx.Client, api_v1: str, email: str, password: str) -> str:
    """Login and return the access token.

    Args:
        client: The HTTP client.
        api_v1: The API v1 base URL.
        email: User email.
        password: User password.

    Returns:
        str: The JWT access token.
    """
    console.print("\n[bold cyan]Step 1: Logging in...[/bold cyan]")
    response = client.post(
        f"{api_v1}/auth/login",
        data={"username": email, "password": password, "grant_type": "password"},
    )

    if response.status_code != 200:
        console.print(f"[bold red]Login failed:[/bold red] {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json()

    table = Table(title="Login Response")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("token_type", data["token_type"])
    table.add_row("access_token", data["access_token"][:40] + "...")
    table.add_row("expires_at", data["expires_at"])
    console.print(table)

    return data["access_token"]


def create_session(client: httpx.Client, api_v1: str, user_token: str) -> tuple[str, str]:
    """Create a new chat session and return session_id and session_token.

    Args:
        client: The HTTP client.
        api_v1: The API v1 base URL.
        user_token: The JWT user access token.

    Returns:
        tuple[str, str]: The session ID and session access token.
    """
    console.print("\n[bold cyan]Step 2: Creating session...[/bold cyan]")
    response = client.post(
        f"{api_v1}/auth/session",
        headers={"Authorization": f"Bearer {user_token}"},
    )

    if response.status_code != 200:
        console.print(f"[bold red]Session creation failed:[/bold red] {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json()
    session_id = data["session_id"]
    session_token = data["token"]["access_token"]

    table = Table(title="Session Created")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("session_id", session_id)
    table.add_row("name", data.get("name", ""))
    table.add_row("session_token", session_token[:40] + "...")
    table.add_row("expires_at", data["token"]["expires_at"])
    console.print(table)

    return session_id, session_token


def research(client: httpx.Client, api_v1: str, session_token: str, message: str) -> list[dict]:
    """Submit a deep research query and return the response messages.

    Args:
        client: The HTTP client.
        api_v1: The API v1 base URL.
        session_token: The JWT session access token.
        message: The research query to submit.

    Returns:
        list[dict]: The response messages.
    """
    console.print(f"\n[bold cyan]Step 3: Submitting research query...[/bold cyan]")
    console.print(Panel(message, title="Research Query"))

    with console.status("[bold green]Researching... this may take a few minutes[/bold green]", spinner="dots"):
        response = client.post(
            f"{api_v1}/deep-research/research",
            headers={"Authorization": f"Bearer {session_token}"},
            json={"messages": [{"role": "user", "content": message}]},
            timeout=600.0,
        )

    if response.status_code != 200:
        console.print(f"[bold red]Research failed:[/bold red] {response.status_code} - {response.text}")
        sys.exit(1)

    data = response.json()
    messages = data["messages"]

    for msg in messages:
        role_color = "blue" if msg["role"] == "user" else "green"
        console.print(Panel(Markdown(msg["content"]), title=f"[{role_color}]{msg['role']}[/{role_color}]"))

    return messages


def research_stream(client: httpx.Client, api_v1: str, session_token: str, message: str) -> str:
    """Submit a deep research query with streaming response.

    Args:
        client: The HTTP client.
        api_v1: The API v1 base URL.
        session_token: The JWT session access token.
        message: The research query to submit.

    Returns:
        str: The full research report content.
    """
    console.print(f"\n[bold cyan]Step 3: Submitting research query (streaming)...[/bold cyan]")
    console.print(Panel(message, title="Research Query"))

    full_content = ""

    with httpx.Client(timeout=600.0) as stream_client:
        with stream_client.stream(
            "POST",
            f"{api_v1}/deep-research/research/stream",
            headers={"Authorization": f"Bearer {session_token}"},
            json={"messages": [{"role": "user", "content": message}]},
        ) as response:
            if response.status_code != 200:
                console.print(
                    f"[bold red]Research stream failed:[/bold red] {response.status_code} - {response.text}"
                )
                sys.exit(1)

            with Live(console=console, refresh_per_second=4) as live:
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue

                    payload = line[len("data: "):]
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("done"):
                        break

                    content = chunk.get("content", "")
                    if content:
                        full_content += content
                        live.update(Panel(Markdown(full_content), title="[green]Research Report[/green]"))

    console.print(Panel(Markdown(full_content), title="[green]Final Research Report[/green]"))
    return full_content


def main() -> None:
    """Run the Deep Research API client flow: login -> create session -> research."""
    args = parse_args()

    base_url = args.base_url.rstrip("/")
    api_v1 = f"{base_url}/api/v1"

    console.print(Panel(f"[bold]Deep Research API Client[/bold]\nTarget: {base_url}", title="LangGraph Deep Research"))

    with httpx.Client() as client:
        # Optional: register user first
        if args.register:
            register_user(client, api_v1, args.email, args.password)

        # Step 1: Login
        user_token = login(client, api_v1, args.email, args.password)

        # Step 2: Create session
        session_id, session_token = create_session(client, api_v1, user_token)

        # Interactive loop
        console.print("\n[bold yellow]Interactive mode[/bold yellow] (type 'quit' to exit)\n")
        while True:
            try:
                user_input = console.input("[bold magenta]Research > [/bold magenta]")
                if user_input.strip().lower() in ("quit", "exit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break
                if not user_input.strip():
                    continue
                if args.stream:
                    research_stream(client, api_v1, session_token, user_input.strip())
                else:
                    research(client, api_v1, session_token, user_input.strip())
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break


if __name__ == "__main__":
    main()
