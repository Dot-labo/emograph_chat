import os
import dotenv
import yaml
from typing import Union, Optional, List, Literal
from pydantic import BaseModel, Field
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager

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
    type: str = Field(..., description="要素のタイプ")


class EmojiElement(BaseElement):
    id: str = Field(..., description="要素のID")
    emoji: str = Field(..., description="絵文字")
    position: Position = Field(..., description="位置")
    size: int = Field(..., description="サイズ")
    font_path: Optional[str] = Field(default=None, description="フォントパス")
    rotation: Optional[int] = Field(default=None, description="回転角度")
    caption: Optional[Caption] = Field(default=None, description="キャプション")


class TextElement(BaseElement):
    id: str = Field(..., description="要素のID")
    content: str = Field(..., description="テキストの内容")
    position: Position = Field(..., description="位置")
    font_size: int = Field(..., description="フォントサイズ")
    color: str = Field(..., description="色")
    font_path: Optional[str] = Field(default=None, description="フォントパス")
    rotation: Optional[int] = Field(default=None, description="回転角度")


class BaseShapeElement(BaseElement):
    id: str = Field(..., description="要素のID")
    color: str = Field(..., description="色")
    thickness: int = Field(..., description="太さ")
    rotation: Optional[int] = Field(default=None, description="回転角度")
    caption: Optional[Caption] = Field(default=None, description="キャプション")


class CircleElement(BaseShapeElement):
    shape: str = Field("circle", description="形状")
    size: CircleSize = Field(..., description="サイズ")
    position: Position = Field(..., description="位置")


class RectangleElement(BaseShapeElement):
    shape: str = Field("rectangle", description="形状")
    size: RectangleSize = Field(..., description="サイズ")
    position: Position = Field(..., description="位置")


class LineElement(BaseShapeElement):
    shape: str = Field("line", description="形状")
    position: LinePosition = Field(..., description="線の位置")


ShapeElement = Union[CircleElement, RectangleElement, LineElement]


class ArrowElement(BaseElement):
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


def load_prompt(file_path: str) -> SystemMessage | AIMessage | HumanMessage:
    with open(file_path, "r") as file:
        prompt_data = yaml.safe_load(file)["prompt"]

    if prompt_data["role"] == "system":
        return SystemMessage(content=prompt_data["content"])
    elif prompt_data["role"] == "assistant":
        return AIMessage(content=prompt_data["content"])
    elif prompt_data["role"] == "user":
        return HumanMessage(content=prompt_data["content"])
    else:
        raise ValueError(f"Invalid role: {prompt_data['role']}")


def main():
    st.title("森羅万象回答大臣Bot")
    st.subheader("━━━教育を、取り戻す。")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    st_callback = StreamlitCallbackHandler(st.container())
    callback_manager = CallbackManager([st_callback])
    system_message = load_prompt("app/prompts/system.yml")

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        streaming=True,
        callback_manager=callback_manager,
        api_key=os.getenv("OPENAI_API_KEY")
    ).with_structured_output(
        OutputSchema
    )

    if prompt := st.chat_input():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # Convert messages to LangChain message format
            messages = []
            messages.append(system_message)
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

            output = llm.invoke(messages)
            ai_response = output.response
            emograph_blueprint_yml = yaml.dump(
                output.emograph_blueprint.model_dump(),
                indent=4
            )
            response = f"{ai_response}\n\n```yaml\n{emograph_blueprint_yml}\n```"

            # Show response
            st.write(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
