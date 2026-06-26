"""
LLM 客户端：使用 Gitee AI 平台生成答案
支持 OpenAI 兼容的 chat/completions 接口，备选本地 Qwen3-4B
"""

import time
from openai import OpenAI
from phase2_rag.config import (
    GITEE_API_BASE, GITEE_API_KEY,
    API_MAX_RETRIES, API_RETRY_DELAY, LLM_MAX_NEW_TOKENS
)

# 答案生成的 prompt 模板
ANSWER_PROMPT_TEMPLATE = (
    "Based on the following context, answer the question briefly. "
    "If the answer is yes/no, answer with yes or no. "
    "If you cannot find the answer, say 'unknown'.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class LLMClient:
    """LLM 答案生成客户端 (Gitee AI API)"""

    def __init__(self, api_key=None, model="Qwen3-4B"):
        self.api_key = api_key or GITEE_API_KEY
        self.model = model
        if not self.api_key:
            raise ValueError("GITEE_API_KEY 未设置")
        self.client = OpenAI(
            base_url=GITEE_API_BASE,
            api_key=self.api_key,
            default_headers={"X-Failover-Enabled": "true"},
        )

    def generate_answer(self, question, context, max_tokens=None):
        """基于上下文生成答案"""
        max_tokens = max_tokens or LLM_MAX_NEW_TOKENS
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)

        for attempt in range(API_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                delay = API_RETRY_DELAY * (2 ** attempt)
                print(f"[LLM] 重试 {attempt+1}/{API_MAX_RETRIES}: {e}")
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(delay)
                else:
                    raise

    def generate_answers_batch(self, questions, contexts, max_tokens=None):
        """批量生成答案"""
        answers = []
        for i, (q, ctx) in enumerate(zip(questions, contexts)):
            answers.append(self.generate_answer(q, ctx, max_tokens))
            if (i + 1) % 10 == 0:
                print(f"[LLM] 进度: {i+1}/{len(questions)}")
        return answers


class LocalLLMClient:
    """本地 LLM 客户端（备选：本地 Qwen3-4B）"""

    def __init__(self, model_path="models/Qwen3-4B"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[LocalLLM] 加载: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            device_map="auto", torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        print("[LocalLLM] 加载完成")

    def generate_answer(self, question, context, max_tokens=128):
        import torch

        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=False,
            )

        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
