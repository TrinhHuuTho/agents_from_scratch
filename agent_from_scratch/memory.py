import json
import sqlite3
from typing import Any, List


class SQLiteMemoryManager:
    """Manages short-term memory using SQLite with support for Thread IDs and Summarization."""

    def __init__(self, db_path: str = "memory.db", max_messages: int = 10):
        self.db_path = db_path
        self.max_messages = max_messages
        self._init_db()


    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT
                )
                """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    thread_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL
                )
                """)
            conn.commit()


    def _dict_to_row(self, thread_id: str, message: dict[str, Any]) -> tuple:
        """Convert a message dictionary to a database row.
        Args:
            thread_id: The ID of the conversation thread.
            message: The message dictionary to convert.
        Returns:
            A tuple representing the database row.
        """
        role = message.get("role")
        content = message.get("content")
        tool_calls = (
            json.dumps(message.get("tool_calls")) if message.get("tool_calls") else None
        )
        tool_call_id = message.get("tool_call_id")
        return (thread_id, role, content, tool_calls, tool_call_id)


    def _row_to_dict(self, row: tuple) -> dict[str, Any]:
        """Convert a database row back to a message dictionary.
        Args:
            row: The database row to convert.
        Returns:
            A dictionary representing the message.
        """
        _, _, role, content, tool_calls_str, tool_call_id = row
        msg = {"role": role}
        if content is not None:
            msg["content"] = content
        if tool_calls_str:
            msg["tool_calls"] = json.loads(tool_calls_str)
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        return msg


    def add_messages(self, thread_id: str, messages: List[dict[str, Any]]):
        """Append messages to a specific thread.
        Args:
            thread_id: The ID of the conversation thread.
            messages: A list of message dictionaries to add.
            Returns:
            None.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            rows = [self._dict_to_row(thread_id, msg) for msg in messages]
            cursor.executemany(
                "INSERT INTO messages (thread_id, role, content, tool_calls, tool_call_id) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()


    def get_messages(self, thread_id: str, system_prompt: str) -> List[dict[str, Any]]:
        """Retrieve the conversation history for a thread, including the system prompt and summary if it exists.

        Args:
            thread_id: The ID of the conversation thread.
            system_prompt: The system prompt to include at the beginning of the messages.
        Returns:
            A list of message dictionaries representing the conversation history.
        """
        messages = [{"role": "system", "content": system_prompt}]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check for existing summary
            cursor.execute(
                "SELECT summary FROM summaries WHERE thread_id = ?", (thread_id,)
            )
            summary_row = cursor.fetchone()
            if summary_row:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Previous Conversation Summary: {summary_row[0]}",
                    }
                )

            # Get actual messages
            cursor.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY id ASC",
                (thread_id,),
            )
            rows = cursor.fetchall()
            messages.extend([self._row_to_dict(row) for row in rows])

        return messages


    def needs_summarization(self, thread_id: str) -> bool:
        """Check if a thread has exceeded the maximum allowed messages.

        Args:
            thread_id: The ID of the conversation thread.
        Returns:
            True if the thread needs summarization, False otherwise.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)
            )
            count = cursor.fetchone()[0]
            return count > self.max_messages


    def get_raw_messages_for_summarization(
        self, thread_id: str
    ) -> List[dict[str, Any]]:
        """Get just the raw user/assistant messages for the summarization prompt.

        Args:
            thread_id: The ID of the conversation thread.
        Returns:
            A list of message dictionaries representing the raw messages.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # We skip tool calls and tool responses to keep the summary prompt cleaner
            cursor.execute(
                "SELECT * FROM messages WHERE thread_id = ? AND (role = 'user' OR role = 'assistant') ORDER BY id ASC",
                (thread_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]


    def save_summary_and_clear(self, thread_id: str, new_summary: str):
        """Save the new summary and delete the old messages to free up context.

        Args:
            thread_id: The ID of the conversation thread.
            new_summary: The new summary to save.
        Returns:
            None.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Upsert the summary
            cursor.execute(
                """
                INSERT INTO summaries (thread_id, summary) 
                VALUES (?, ?) 
                ON CONFLICT(thread_id) DO UPDATE SET summary=excluded.summary
                """,
                (thread_id, new_summary),
            )

            # Delete old messages
            cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            conn.commit()
