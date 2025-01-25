
import asyncio
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from models.emograph import OutputSchema


#TODO: 本機能はLangChainの機能としてネイティブに実装されている可能性があるので、その場合は差し替えを検討
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


#TODO: 本機能はLangChainの機能としてネイティブに実装されている可能性があるので、その場合は差し替えを検討
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
