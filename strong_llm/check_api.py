"""
API 状态快速检查

用法:
  python check_api.py              # 检查当前 config 中的默认 provider
  python check_api.py zhipu        # 检查智谱
  python check_api.py deepseek     # 检查 DeepSeek
  python check_api.py all          # 检查所有已配置的 provider
"""

import sys
sys.path.insert(0, ".")
import config


def check_provider(provider: str) -> bool:
    """检查指定 provider 的 API 可用性，返回是否可用。"""
    cfg = config.LLM_PROVIDERS.get(provider)
    if not cfg:
        print(f"  [{provider}] 未配置")
        return False

    api_key = cfg.get("api_key", "")
    model = cfg.get("model", "")
    if not api_key:
        print(f"  [{provider}] 未设置 API Key (.env 中缺少对应 KEY)")
        return False

    print(f"  [{provider}] 模型: {model}, Key: {api_key[:8]}...{api_key[-4:]}")

    if provider == "zhipu":
        return _check_zhipu(api_key, model)
    else:
        return _check_openai_compat(provider, cfg)


def _check_zhipu(api_key: str, model: str) -> bool:
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)

    # 免费模型
    try:
        client.chat.completions.create(
            model="glm-4-flash", messages=[{"role": "user", "content": "hi"}], max_tokens=1
        )
        print(f"    glm-4-flash (免费): OK")
    except Exception as e:
        print(f"    glm-4-flash (免费): FAIL - {str(e)[:100]}")

    # 付费模型
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=1
        )
        print(f"    {model} (付费): OK")
        return True
    except Exception as e:
        if "1113" in str(e):
            print(f"    {model} (付费): 余额不足!")
        else:
            print(f"    {model} (付费): FAIL - {str(e)[:100]}")
        return False


def _check_openai_compat(provider: str, cfg: dict) -> bool:
    from openai import OpenAI
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url"))
    model = cfg["model"]

    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=1
        )
        print(f"    {model}: OK")
        return True
    except Exception as e:
        err = str(e)[:150]
        if "402" in err or "insufficient" in err.lower() or "quota" in err.lower():
            print(f"    {model}: 余额不足!")
        else:
            print(f"    {model}: FAIL - {err}")
        return False


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else config.LLM_PROVIDER

    print("=" * 50)
    print("API 状态检查")
    print("=" * 50)

    if target == "all":
        providers = [p for p in config.LLM_PROVIDERS if config.LLM_PROVIDERS[p].get("api_key")]
    else:
        providers = [target]

    results = {}
    for p in providers:
        print()
        ok = check_provider(p)
        results[p] = ok

    print()
    print("-" * 50)
    for p, ok in results.items():
        status = "可用" if ok else "不可用"
        print(f"  {p}: {status}")
    print("-" * 50)

    current = config.LLM_PROVIDER
    if current in results and not results[current]:
        available = [p for p, ok in results.items() if ok]
        if available:
            print(f"\n提示: 当前 LLM_PROVIDER='{current}' 不可用")
            print(f"  可选: 修改 config.py LLM_PROVIDER = \"{available[0]}\"")
            print(f"  或: python run.py --mode compare --strategies E0 --provider {available[0]}")


if __name__ == "__main__":
    main()
