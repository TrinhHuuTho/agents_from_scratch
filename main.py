import logging

from agent_from_scratch.agent import SimpleAgent
from agent_from_scratch.chat_models import ChatModel
from agent_from_scratch.config import load_settings


def main() -> None:
    """Run the demo agent from the command line.

    Returns:
        None.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        settings = load_settings()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return

    client = ChatModel(
        api_key=settings.openrouter_api_key,
        model=settings.model,
        base_url=settings.base_url,
        app_name=settings.app_name,
    )

    agent = SimpleAgent(
        client=client,
        system_prompt=settings.system_prompt,
        skills=['./skills/langgraph-docs'],
        tools=['ls_tool', 'read_file', 'write_file', 'edit_file', 'fetch_url'],
        max_iterations=settings.max_iterations,
    )


    result1 = agent.invoke(
        {
            "thread_id": "user_123",
            "messages": [
                {
                    "role": "user",
                    "content": "Chào bạn, dựa vào tài liệu từ langgraph-docs. Hãy hướng dẫn tôi cách xây dựng skills cho agent from scratch nhé?",
                },
            ]
        }
    )
    final_message1 = result1.get("messages", [])[-1] if result1.get("messages") else None
    content1 = final_message1.get("content") if isinstance(final_message1, dict) else None
    logger.info("User: Chào bạn, dựa vào tài liệu từ langgraph-docs. Hãy hướng dẫn tôi cách xây dựng skills cho agent from scratch nhé?")
    logger.info("Agent: %s", content1)


if __name__ == "__main__":
    main()
