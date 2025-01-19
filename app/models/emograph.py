from typing import Union, Optional, List, Literal
from pydantic import BaseModel, Field


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
