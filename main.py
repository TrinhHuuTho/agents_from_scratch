import json
import logging
from urllib import response
from dataclasses import dataclass

from agent_from_scratch.agent import SimpleAgent
from agent_from_scratch.chat_models import ChatModel
from agent_from_scratch.config import load_settings
from agent_from_scratch.runtime import Runtime


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
        sensitive_tools=['write_file', 'edit_file', 'fetch_url']
    )

    content = input("How can I help you today? ").strip()
    inputs = {
        "thread_id": "user_1",
            "messages": [
                {
                    "role": "user",
                    "content": content,
                },
            ]
    }
    
    # Define a simple context class if not provided by the library
    @dataclass
    class MyContext:
        user_id: str

    custom_rt = Runtime(context=MyContext(user_id="test_user"))

    response = agent.invoke(inputs, runtime=custom_rt)
    
    while response["status"] == "requires_action":
        print(f"\n🛑 HITL INTERRUPT: {response['message']}")
        raw_args = response['tool_call']['function']['arguments']
        print(f"Arguments found: {raw_args}")
        
        # Simulate a human review process (can be a CLI input or UI interaction)
        choice = input("Approve action? (yes / no / edit or typing your own message): ").strip().lower()
        
        if choice == "yes":
            user_response = {"decision": "approve", "tool_call": response["tool_call"]}
            
        elif choice == "edit":
            raw_args = response["tool_call"]["function"]["arguments"]
            parsed_args = json.loads(raw_args)
            
            print(f"Current arguments: {parsed_args}")
            
            if "url" in parsed_args:
                new_url = input(f"Enter new URL (press Enter to keep '{parsed_args['url']}'): ").strip()
                if new_url:
                    parsed_args["url"] = new_url
                    
            if "content" in parsed_args:
                parsed_args["content"] = parsed_args["content"] + "\n\n[Reviewed and appended by Human Supervisor]"
            elif "text" in parsed_args:
                parsed_args["text"] = parsed_args["text"] + "\n\n[Reviewed and appended by Human Supervisor]"
                
            user_response = {
                "decision": "edit", 
                "tool_call": response["tool_call"],
                "edited_args": parsed_args
            }
            
        elif choice == "no":
            user_response = {"decision": "reject", "tool_call": response["tool_call"]}
        
        else:
            user_response = {"decision": "respond", "tool_call": response["tool_call"], "human_message": choice}
              
        # Resume agent execution passing the human response structure back in
        response = agent.invoke(inputs, user_response=user_response)
        
    print("\nAgent Completed Execution:")
    print(response["final_message"]["content"])

    

if __name__ == "__main__":
    main()
