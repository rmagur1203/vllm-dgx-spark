"""
Patch vLLM's DeltaMessage to output 'reasoning_content' instead of 'reasoning'.

LiteLLM (hosted_vllm provider) expects OpenAI-standard 'reasoning_content' field,
but vLLM 0.17.x outputs 'reasoning'. This monkey-patch fixes the JSON serialization
without modifying vLLM source files.
"""

def apply():
    """Add reasoning_content property that aliases reasoning field."""
    try:
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage
        
        # Store original model_fields
        original_field = DeltaMessage.model_fields.get('reasoning')
        if original_field is None:
            return  # Already patched or field doesn't exist
        
        # Rebuild the model with reasoning_content field name
        # We do this by modifying the serialization output
        original_dump = DeltaMessage.model_dump_json
        
        def patched_dump_json(self, **kwargs):
            result = original_dump(self, **kwargs)
            # Replace "reasoning" with "reasoning_content" in JSON output
            return result.replace('"reasoning":', '"reasoning_content":')
        
        DeltaMessage.model_dump_json = patched_dump_json
        print("[reasoning patch] DeltaMessage.reasoning → reasoning_content in JSON output")
    except Exception as e:
        print(f"[reasoning patch] Warning: {e}")

apply()
