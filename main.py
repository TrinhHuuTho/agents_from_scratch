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
        tools=['ls_tool', 'read_file', 'write_file', 'edit_file'],
        max_iterations=settings.max_iterations,
    )

    logger.info("--- Lượt 1 ---")
    result1 = agent.invoke(
        {
            "thread_id": "user_123",
            "messages": [
                {
                    "role": "user",
                    "content": "Chào bạn, tôi tên là Thor.",
                },
            ]
        }
    )
    final_message1 = result1.get("messages", [])[-1] if result1.get("messages") else None
    content1 = final_message1.get("content") if isinstance(final_message1, dict) else None
    logger.info("User: Chào bạn, tôi tên là Thor.")
    logger.info("Agent: %s", content1)

    logger.info("--- Lượt 2 ---")
    result2 = agent.invoke(
        {
            "thread_id": "user_123",
            "messages": [
                {
                    "role": "user",
                    "content": "Bạn có nhớ tôi tên là gì không?",
                },
            ]
        }
    )
    final_message2 = result2.get("messages", [])[-1] if result2.get("messages") else None
    content2 = final_message2.get("content") if isinstance(final_message2, dict) else None
    logger.info("User: Bạn có nhớ tôi tên là gì không?")
    logger.info("Agent: %s", content2)

    logger.info("--- Lượt 3 (Pushing past max messages) ---")
    result3 = agent.invoke(
        {
            "thread_id": "user_123",
            "messages": [
                {
                    "role": "user",
                    "content": "Sở thích của tôi là lập trình AI và chơi game.",
                },
            ]
        }
    )
    final_message3 = result3.get("messages", [])[-1] if result3.get("messages") else None
    content3 = final_message3.get("content") if isinstance(final_message3, dict) else None
    logger.info("User: Sở thích của tôi là lập trình AI và chơi game.")
    logger.info("Agent: %s", content3)

    logger.info("--- Lượt 4 (Testing Summarized Memory) ---")
    result4 = agent.invoke(
        {
            "thread_id": "user_123",
            "messages": [
                {
                    "role": "user",
                    "content": "Tổng hợp lại, tôi là ai và có sở thích gì?",
                },
            ]
        }
    )
    final_message4 = result4.get("messages", [])[-1] if result4.get("messages") else None
    content4 = final_message4.get("content") if isinstance(final_message4, dict) else None
    logger.info("User: Tổng hợp lại, tôi là ai và có sở thích gì?")
    logger.info("Agent: %s", content4)


if __name__ == "__main__":
    main()
