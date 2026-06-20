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
        sensitive_tools=['write_file', 'edit_file']
    )

    inputs = {
        "thread_id": "user_456",
            "messages": [
                {
                    "role": "user",
                    "content": 'Write for me a file with content discussing about "AI is more intelligent than ever before. Now, they can write code, translate languages in real time, and even create art. People do not need to learn programing or foreign languages anymore. Write out in vietnamese"',
                },
            ]
    }
    response = agent.invoke(inputs)
    

    if response["status"] == "requires_action":
        print(f"\n🛑 HITL INTERRUPT: {response['message']}")
        print(f"Arguments found: {response['tool_call']['function']['arguments']}")
        
        # Simulate a human review process (can be a CLI input or UI interaction)
        choice = input("Approve action? (yes / no / edit or typing your own message): ").strip().lower()
        
        if choice == "yes":
            user_response = {"decision": "approve", "tool_call": response["tool_call"]}
        elif choice == "edit":
            user_response = {
                "decision": "edit", 
                "tool_call": response["tool_call"],
                "edited_args": {"filename": "document.txt", "content": "Hello World from human editor!"}
            }
        elif choice == "no":
            user_response = {"decision": "reject", "tool_call": response["tool_call"]}
        else:
            user_response = {"decision": "respond", "tool_call": response["tool_call"], "human_message": choice}

        # Resume agent execution passing the human response structure back in
        final_result = agent.invoke(inputs, user_response=user_response)
        print("\nAgent Completed Execution:")
        print(final_result["final_message"]["content"])
    else:
        print(response["final_message"]["content"])


if __name__ == "__main__":
    main()
