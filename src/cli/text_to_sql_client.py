"""Script to interact with the Text-to-SQL API: login, create session, and submit natural language queries.

Usage:
    python src/cli/text_to_sql_client.py --email user@example.com --password yourpassword
    python src/cli/text_to_sql_client.py --email user@example.com --password yourpassword --register
"""

import argparse
import sys

import httpx
from rich.console import Console
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
    parser = argparse.ArgumentParser(description="Text-to-SQL API client for natural language database queries.")
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--email", required=True, help="User email for login")
    parser.add_argument("--password", required=True, help="User password for login")
    parser.add_argument("--register", action="store_true", help="Register a new user before login")
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
    console.print("\n[bold cyan]Registering user...[/bold cyan]")
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


def query(client: httpx.Client, api_v1: str, session_token: str, user_query: str) -> list[dict]:
    """Submit a natural language query and return the SQL agent response.

    Args:
        client: The HTTP client.
        api_v1: The API v1 base URL.
        session_token: The JWT session access token.
        user_query: The natural language question about the database.

    Returns:
        list[dict]: The response messages from the agent.
    """
    console.print(f"\n[bold cyan]Step 3: Submitting query...[/bold cyan]")
    console.print(Panel(user_query, title="SQL Query"))

    with console.status("[bold green]Querying database... this may take a moment[/bold green]", spinner="dots"):
        response = client.post(
            f"{api_v1}/text-to-sql/query",
            headers={"Authorization": f"Bearer {session_token}"},
            json={"query": user_query},
            timeout=300.0,
        )

    if response.status_code != 200:
        console.print(f"[bold red]Query failed:[/bold red] {response.status_code} - {response.text}")
        return []

    data = response.json()
    messages = data["messages"]

    for msg in messages:
        role_color = "blue" if msg["role"] == "user" else "green"
        console.print(Panel(Markdown(msg["content"]), title=f"[{role_color}]{msg['role']}[/{role_color}]"))

    return messages


def main() -> None:
    """Run the Text-to-SQL API client flow: login -> create session -> query loop."""
    args = parse_args()

    base_url = args.base_url.rstrip("/")
    api_v1 = f"{base_url}/api/v1"

    console.print(
        Panel(
            "[bold]Text-to-SQL API Client[/bold]\n"
            f"Target: {base_url}\n"
            "Ask natural language questions about your database.",
            title="LangGraph Text-to-SQL",
        )
    )

    with httpx.Client() as client:
        if args.register:
            register_user(client, api_v1, args.email, args.password)

        user_token = login(client, api_v1, args.email, args.password)
        session_id, session_token = create_session(client, api_v1, user_token)

        console.print("\n[bold yellow]Interactive mode[/bold yellow] (type 'quit' to exit)")
        console.print("[dim]Example: How many customers are from Canada?[/dim]\n")

        while True:
            try:
                user_input = console.input("[bold magenta]SQL > [/bold magenta]")
                if user_input.strip().lower() in ("quit", "exit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break
                if not user_input.strip():
                    continue
                query(client, api_v1, session_token, user_input.strip())
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break


if __name__ == "__main__":
    main()
