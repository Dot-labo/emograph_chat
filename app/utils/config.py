import yaml
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage


def load_system_config(file_path: str) -> dict:
    """
    システム設定を読み込む

    Args:
        file_path (str): ファイルパス

    Returns:
        dict: システム設定
    """
    with open(file_path, "r", encoding="utf-8") as file:
        system_config = yaml.safe_load(file)
    return system_config


def load_prompt(file_path: str) -> SystemMessage | AIMessage | HumanMessage:
    """
    プロンプトを読み込む

    Args:
        file_path (str): ファイルパス

    Returns:
        SystemMessage | AIMessage | HumanMessage: プロンプト
    """
    with open(file_path, "r", encoding="utf-8") as file:
        prompt_data = yaml.safe_load(file)["prompt"]

    if prompt_data["role"] == "system":
        return SystemMessage(content=prompt_data["content"])
    elif prompt_data["role"] == "assistant":
        return AIMessage(content=prompt_data["content"])
    elif prompt_data["role"] == "user":
        return HumanMessage(content=prompt_data["content"])
    else:
        raise ValueError(f"Invalid role: {prompt_data['role']}")
