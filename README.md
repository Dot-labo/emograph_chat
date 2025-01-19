# emograph_chat

## 設定

- env

    .env.templateをコピーして.envを作成してください。
    現状、記入すべき項目は`OPENAI_API_KEY`のみです。

    ``` bash
    cp .env.template .env
    ```

- config

    config直下の各種設定ファイルをコピーし、必要に応じて編集してください

    ``` bash
    cp config/prompt_template.yml config/prompt.yml
    cp config/system_template.yml config/system.yml
    ```

## 起動方法

``` bash
uv run streamlit run app/main.py
```
