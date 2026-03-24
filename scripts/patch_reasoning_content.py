#!/usr/bin/env python3
"""
Patch vLLM DeltaMessage to include 'reasoning_content' field alongside 'reasoning'.

vLLM 0.17.x outputs streaming delta as {"reasoning": "..."} but LiteLLM's
hosted_vllm provider expects OpenAI-standard {"reasoning_content": "..."}.
GPT-OSS on vLLM 0.13 outputs both fields.

This script patches the protocol file at container startup so that
DeltaMessage auto-mirrors reasoning → reasoning_content via model_post_init.
"""

PROTOCOL = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/engine/protocol.py"

try:
    code = open(PROTOCOL).read()

    if "reasoning_content" in code:
        print("[reasoning_content] Already patched")
    else:
        # Replace DeltaMessage class to add reasoning_content with auto-sync
        old = """class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)"""

        new = """class DeltaMessage(OpenAIBaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)

    def model_post_init(self, __context):
        # Mirror reasoning → reasoning_content for LiteLLM compatibility
        if self.reasoning is not None and self.reasoning_content is None:
            self.reasoning_content = self.reasoning"""

        if old in code:
            code = code.replace(old, new)
            open(PROTOCOL, 'w').write(code)
            print("[reasoning_content] Patched DeltaMessage with auto-sync")
        else:
            print("[reasoning_content] WARNING: DeltaMessage pattern not found")
except Exception as e:
    print(f"[reasoning_content] Error: {e}")
