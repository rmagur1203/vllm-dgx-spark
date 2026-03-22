from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("super_v3")
class SuperV3ReasoningParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )

        # Nemotron can spend the whole decode budget inside <think> without
        # ever emitting final content. In that case, surface the reasoning
        # text as content so OpenAI-compatible clients do not receive null.
        if final_content is None and reasoning_content:
            return None, reasoning_content

        return reasoning_content, final_content
