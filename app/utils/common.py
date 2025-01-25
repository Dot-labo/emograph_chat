import yaml


def format_yaml_for_display(yaml_content: str) -> str:
    """
    YAMLを表示用に整形する
    - マルチバイト文字（日本語、絵文字）を正しく表示
    - インデントと改行を整える
    - キーの順序を保持
    """
    yaml_dict = yaml.safe_load(yaml_content)
    return yaml.dump(yaml_dict, allow_unicode=True, default_flow_style=False, sort_keys=False)
