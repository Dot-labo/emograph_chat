import os
import dotenv
import yaml
from typing import Union, Optional, List, Literal
from pydantic import BaseModel, Field
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks import StreamlitCallbackHandler
#from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from emograph import Builder
import asyncio

dotenv.load_dotenv()


class Position(BaseModel):
    x: int = Field(..., description="位置のX座標")
    y: int = Field(..., description="位置のY座標")


class Caption(BaseModel):
    content: str = Field(..., description="キャプションの内容")
    position: Position = Field(..., description="キャプションの位置")
    font_size: int = Field(..., description="キャプションのフォントサイズ")
    color: str = Field(..., description="キャプションの色")


class LinePosition(BaseModel):
    start: Position = Field(..., description="線の始点")
    end: Position = Field(..., description="線の終点")


class ArrowPosition(BaseModel):
    start: Position = Field(..., description="矢印の始点")
    end: Position = Field(..., description="矢印の終点")


class CircleSize(BaseModel):
    rx: int = Field(..., description="円の横幅")
    ry: int = Field(..., description="円の縦幅")


class RectangleSize(BaseModel):
    width: int = Field(..., description="矩形の横幅")
    height: int = Field(..., description="矩形の縦幅")


class BaseElement(BaseModel):
    type: Literal["emoji", "arrow", "text", "shape"] = Field(..., description="要素のタイプ")


class EmojiElement(BaseElement):
    type: Literal["emoji"] = Field(..., description="要素のタイプ")
    id: str = Field(..., description="要素のID")
    emoji: str = Field(..., description="絵文字")
    position: Position = Field(..., description="位置")
    size: int = Field(..., description="サイズ")
    font_path: Optional[str] = Field(default=None, description="フォントパス")
    rotation: Optional[int] = Field(default=None, description="回転角度")
    caption: Optional[Caption] = Field(default=None, description="キャプション")


class TextElement(BaseElement):
    type: Literal["text"] = Field(..., description="要素のタイプ")
    id: str = Field(..., description="要素のID")
    content: str = Field(..., description="テキストの内容")
    position: Position = Field(..., description="位置")
    font_size: int = Field(..., description="フォントサイズ")
    color: str = Field(..., description="色")
    font_path: Optional[str] = Field(default=None, description="フォントパス")
    rotation: Optional[int] = Field(default=None, description="回転角度")


class ShapeElement(BaseElement):
    type: Literal["shape"] = Field(..., description="要素のタイプ")
    shape: Literal["circle", "rectangle", "line"] = Field(..., description="形状")
    id: str = Field(..., description="要素のID")
    color: str = Field(..., description="色")
    thickness: int = Field(..., description="太さ")
    size: Optional[Union[CircleSize, RectangleSize]] = Field(default=None, description="サイズ（円の場合は円の直径、矩形の場合は矩形の横幅と縦幅、線の場合はNone）")
    position: Optional[Union[Position, LinePosition]] = Field(default=None, description="位置（円と矩形の場合はPosition、線の場合はLinePosition）")
    rotation: Optional[int] = Field(default=None, description="回転角度")
    caption: Optional[Caption] = Field(default=None, description="キャプション")


class ArrowElement(BaseElement):
    type: Literal["arrow"] = Field(..., description="要素のタイプ")
    start_id: str = Field(..., description="始点のID")
    end_id: str = Field(..., description="終点のID")
    color: str = Field(..., description="矢印の色")
    position: ArrowPosition = Field(..., description="矢印の位置")
    thickness: int = Field(..., description="矢印の太さ")


EmographElement = Union[EmojiElement, ArrowElement, TextElement, ShapeElement]


class EmographBlueprint(BaseModel):
    width: int = Field(..., description="生成画像の横幅")
    height: int = Field(..., description="生成画像の縦幅")
    background_color: str = Field(..., description="背景色")
    text_font_path: Optional[str] = Field(default=None, description="テキストのフォントパス")
    emoji_font_path: Optional[str] = Field(default=None, description="絵文字のフォントパス")
    elements: List[EmographElement] = Field(..., description="グラフの構成要素")


class OutputSchema(BaseModel):
    response: str = Field(..., description="質問に対する解答")
    emograph_blueprint: EmographBlueprint = Field(..., description="解答に伴う心的描画の設計図")


def load_system_config(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        system_config = yaml.safe_load(file)
    return system_config


def load_prompt(file_path: str) -> SystemMessage | AIMessage | HumanMessage:
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


async def main():
    system_config = load_system_config("config/system.yml")
    # ページの名前
    st.set_page_config(page_title=system_config["title"], page_icon=":material/emoji_objects:")
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
                    with st.expander("設計図を表示"):
                        st.code(message["additional"]["yaml"], language="yaml")
                if "image" in message["additional"]:
                    st.image(message["additional"]["image"])

    st_callback = StreamlitCallbackHandler(st.container())
    callback_manager = CallbackManager([st_callback])
    system_message = load_prompt("config/prompt.yml")

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        streaming=True,
        callback_manager=callback_manager,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(
        OutputSchema,
        method="json_schema"
    )

    builder = Builder()

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

                # 複数の応答を並行生成
                async def generate_multiple_responses():
                    tasks = [llm.ainvoke(messages) for _ in range(3)]  # 3つの応答を生成
                    outputs = await asyncio.gather(*tasks)
                    return outputs

                outputs = await generate_multiple_responses()
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
                    st.code(emograph_blueprint_yml, language="yaml")

                # Show image
                emograph_image = builder.get_generate_image(emograph_blueprint_dict)
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
