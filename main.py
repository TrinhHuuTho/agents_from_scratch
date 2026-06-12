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

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content":
                    """Thực hiện các bước sau:
                    1) Tạo file 'hello.txt' với nội dung lorem ipsum. 
                    2) Đọc nội dung file 'hello.txt' và trả lời tôi đọc được gì.
                    3) Liệt kê các file trong thư mục workspace và trả lời tôi có bao nhiêu file.
                    4) Sửa file 'hello.txt', thay nội dung thành 'hello world'.",
                    """
                },
            ]
        }
    )

    final_message = result.get("messages", [])[-1] if result.get("messages") else None
    content = final_message.get("content") if isinstance(final_message, dict) else None
    if content:
        logger.info(content)
    else:
        logger.info("No final content available; final_message=%s", final_message)


if __name__ == "__main__":
    main()
