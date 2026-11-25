"""
统一 LLM 调用层

支持多种 LLM Provider（智谱 / DeepSeek / OpenAI / Qwen），
自动记录 token 用量和延迟。
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import config


@dataclass
class LLMResponse:
    """LLM 调用响应"""
    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""


@dataclass
class TokenStats:
    """累计 token 统计"""
    total_input: int = 0
    total_output: int = 0
    total_calls: int = 0
    total_latency_ms: float = 0.0

    def update(self, resp: LLMResponse):
        self.total_input += resp.input_tokens
        self.total_output += resp.output_tokens
        self.total_calls += 1
        self.total_latency_ms += resp.latency_ms

    def summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "total_calls": self.total_calls,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "avg_latency_ms": round(self.total_latency_ms / max(self.total_calls, 1), 1),
        }


class LLMClient:
    """
    统一 LLM 调用客户端

    使用方法:
        client = LLMClient(provider="zhipu")
        resp = client.call("请分析这篇文章...")
        print(resp.content, resp.input_tokens)
    """

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.provider = provider or config.LLM_PROVIDER
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS

        # 从注册表获取默认配置
        provider_cfg = config.LLM_PROVIDERS.get(self.provider, {})
        self.model = model or provider_cfg.get("model", "")
        self.api_key = api_key or provider_cfg.get("api_key", "")
        self.base_url = base_url or provider_cfg.get("base_url")

        # 累计统计
        self.stats = TokenStats()

        # 初始化底层客户端
        self._client = None
        self._init_client()

    def _init_client(self):
        """根据 provider 初始化底层 SDK 客户端"""
        if self.provider == "zhipu":
            from zhipuai import ZhipuAI
            self._client = ZhipuAI(api_key=self.api_key)

        elif self.provider in ("deepseek", "openai", "qwen"):
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            raise ValueError(f"不支持的 LLM Provider: {self.provider}")

    def call(
        self,
        content: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> LLMResponse:
        """
        发送单轮对话请求

        Args:
            content: 用户消息内容
            system_prompt: 系统提示词（可选）
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 token 数

        Returns:
            LLMResponse 包含生成内容和 token 统计
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        return self.call_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def call_messages(
        self,
        messages: List[Dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> LLMResponse:
        """
        发送消息列表请求

        Args:
            messages: [{"role": "user", "content": "..."}]
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 token 数

        Returns:
            LLMResponse
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        max_retries = config.LLM_MAX_RETRIES
        start_time = time.time()
        response = None

        for attempt in range(max_retries + 1):
            try:
                if self.provider == "zhipu":
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=tokens,
                        temperature=temp,
                        top_p=0.15,
                        response_format={"type": "json_object"},
                    )
                else:
                    # OpenAI 兼容接口（DeepSeek / OpenAI / Qwen）
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=tokens,
                        temperature=temp,
                        response_format={"type": "json_object"},
                    )
                break  # 成功则跳出重试循环
            except Exception as e:
                err_str = str(e)
                is_retryable = any(k in err_str for k in ("429", "1302", "rate", "Rate", "Connection", "connection", "timeout", "Timeout", "503", "502"))
                if is_retryable and attempt < max_retries:
                    is_rate_limit = any(k in err_str for k in ("429", "1302"))
                    wait = min(2 ** attempt * (30 if is_rate_limit else 5), 180)  # 429: 30s,60s,120s,180s; 其他: 5s,10s,20s
                    print(f"  [LLM] 请求失败 (attempt {attempt+1}/{max_retries+1}), {wait}s 后重试: {err_str[:80]}")
                    time.sleep(wait)
                else:
                    raise  # 不可重试或已耗尽重试次数

        latency = (time.time() - start_time) * 1000  # ms

        # 解析响应
        resp = LLMResponse(
            content=response.choices[0].message.content or "",
            input_tokens=getattr(response.usage, "prompt_tokens", 0),
            output_tokens=getattr(response.usage, "completion_tokens", 0),
            total_tokens=getattr(response.usage, "total_tokens", 0),
            latency_ms=round(latency, 1),
            model=self.model,
            provider=self.provider,
        )

        # 更新统计
        self.stats.update(resp)
        return resp

    def reset_stats(self):
        """重置累计统计"""
        self.stats = TokenStats()

    def get_stats(self) -> dict:
        """获取累计统计"""
        return self.stats.summary()
