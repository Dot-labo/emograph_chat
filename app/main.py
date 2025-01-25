import os
import dotenv
import yaml
import asyncio
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
#from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from emograph import Builder as EmographBuilder
from utils.config import load_system_config, load_prompt
from models.emograph import OutputSchema

dotenv.load_dotenv()


# 複数の応答を並行生成
async def agenerate_with_retry(
    llm: ChatOpenAI,
    messages: list[SystemMessage | AIMessage | HumanMessage],
    max_retries: int = 3,
    delay: float = 1.0
) -> OutputSchema:
    """
    リトライ機能付きの応答生成

    Args:
        llm (ChatOpenAI): 言語モデル
        messages (list[SystemMessage | AIMessage | HumanMessage]): メッセージ
        max_retries (int): 最大リトライ回数
        delay (float): リトライ間の待機時間（秒）

    Returns:
        OutputSchema: 応答
    """
    for attempt in range(max_retries):
        try:
            return await llm.ainvoke(messages)
        except Exception as e:
            if attempt == max_retries - 1:  # 最後の試行で失敗した場合
                raise e
            await asyncio.sleep(delay * (attempt + 1))  # 指数バックオフ

async def agenerate_multiple_responses(
    llm: ChatOpenAI,
    messages: list[SystemMessage | AIMessage | HumanMessage],
    parallel_count: int = 3
) -> list[OutputSchema]:
    """
    複数の応答を並行生成

    Args:
        llm (ChatOpenAI): 言語モデル
        messages (list[SystemMessage | AIMessage | HumanMessage]): メッセージ
        parallel_count (int): 並行生成数

    Returns:
        list[OutputSchema]: 応答のリスト
    """
    tasks = [agenerate_with_retry(llm, messages) for _ in range(parallel_count)]
    outputs = await asyncio.gather(*tasks, return_exceptions=True)
    
    # エラーをフィルタリング
    valid_outputs = [output for output in outputs if not isinstance(output, Exception)]
    
    return valid_outputs


def convert_yaml_to_readable(yaml_content: str) -> str:
    """
    YAMLをマルチバイト対応の形式に変換
    """
    yaml_dict = yaml.safe_load(yaml_content)
    return yaml.dump(yaml_dict, allow_unicode=True, default_flow_style=False, sort_keys=False)


async def main():
    # システム設定を読み込む
    system_config = load_system_config("config/system.yml")

    # ページの設定
    st.set_page_config(
        page_title=system_config["title"],
        page_icon=":material/emoji_objects:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=system_config["menu_items"]
    )

    # UI
    st.title(system_config["ui"]["title"])
    st.subheader(system_config["ui"]["subheader"])

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_response" not in st.session_state:
        st.session_state.selected_response = False
    if "current_outputs" not in st.session_state:
        st.session_state.current_outputs = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "additional" in message:
                if "yaml" in message["additional"]:
                    with st.expander("設計図を表示・編集"):
                        # YAMLをマルチバイト対応の形式に変換
                        readable_yaml = convert_yaml_to_readable(message["additional"]["yaml"])

                        message_id = id(message)
                        edited_yaml = st.text_area(
                            "blueprint.yml",
                            value=readable_yaml,
                            height=300,
                            key=f"yaml_editor_{message_id}_{len(st.session_state.messages)}"
                        )
                        # 編集内容を保存
                        if edited_yaml != message["additional"]["yaml"]:
                            # YAMLから画像を再生成
                            try:
                                yaml_dict = yaml.safe_load(edited_yaml)
                                emograph_builder = EmographBuilder()
                                new_image = emograph_builder.get_generate_image(yaml_dict)
                                message["additional"]["yaml"] = edited_yaml
                                message["additional"]["image"] = new_image
                                st.rerun()  # 画面を更新して新しい画像を表示
                            except Exception as e:
                                st.error(f"設計図の形式が正しくありません: {str(e)}")

                if "image" in message["additional"]:
                    st.image(message["additional"]["image"])

    # プロンプトを読み込む
    system_message = load_prompt("config/prompt.yml")

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        streaming=True,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(
        OutputSchema,
        method="json_schema"
    )

    # Emographのビルダーを初期化
    emograph_builder = EmographBuilder()

    # 新しい入力があった場合
    if prompt := st.chat_input("何を表現したいですか？"):
        with st.chat_message("user"):
            st.write(prompt)

        # ユーザーの入力を追加
        st.session_state.messages.append({"role": "user", "content": prompt, "additional": []})
        st.session_state.selected_response = False

        # 応答を生成
        with st.chat_message("assistant"):
            with st.spinner("考え中..."):
                messages = []
                messages.append(system_message)
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(
                            AIMessage(
                                content=f"AI: {msg['content']}\n\n```yaml\n{msg['additional']['yaml']}\n```"
                            )
                        )

                outputs = await agenerate_multiple_responses(llm, messages, parallel_count=3)
                if outputs and len(outputs) > 0:
                    st.session_state.current_outputs = outputs

        st.rerun()  # 新しい状態で画面を更新

    # 応答が生成されていて、まだ選択されていない場合に選択肢を表示
    if ("current_outputs" in st.session_state and 
        isinstance(st.session_state.current_outputs, list) and 
        len(st.session_state.current_outputs) > 0 and 
        not st.session_state.get("selected_response", False)):

        st.write("以下の案から選んでください：")
        cols = st.columns(len(st.session_state.current_outputs))

        for i, output in enumerate(st.session_state.current_outputs):
            with cols[i]:
                st.write(f"案 {i+1}")
                ai_response = output.response
                emograph_blueprint_dict = output.emograph_blueprint.model_dump()
                emograph_blueprint_yml = yaml.dump(
                    emograph_blueprint_dict,
                    indent=4
                )
                st.write(ai_response)
                with st.expander("設計図を表示"):
                    # YAMLをマルチバイト対応の形式に変換
                    readable_yaml = convert_yaml_to_readable(emograph_blueprint_yml)
                    st.code(readable_yaml, language="yaml")

                # Show image
                emograph_image = emograph_builder.get_generate_image(emograph_blueprint_dict)
                st.image(emograph_image)

                # 選択ボタン
                if st.button("この案を採用", key=f"select_btn_{i}"):
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": ai_response,
                            "additional": {
                                "yaml": emograph_blueprint_yml,
                                "image": emograph_image,
                            }
                        }
                    )
                    st.session_state.selected_response = True
                    st.session_state.current_outputs = None
                    st.rerun()


if __name__ == "__main__":
    asyncio.run(main())
