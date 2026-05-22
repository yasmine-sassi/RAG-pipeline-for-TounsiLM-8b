import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_ID = "alabenayed/TounsiLM-8b"

_PROMPT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n"
    "{system}\n"
    "<</SYS>>\n\n"
    "{user} [/INST]"
)

_DEFAULT_SYSTEM = (
    "أنت مساعد ذكي متخصص في اللهجة التونسية والثقافة التونسية. "
    "أجب بدقة ووضوح. إذا أُعطيت معلومات من قاعدة المعرفة، استخدمها."
)


class TounsiLM:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
    ):
        self.model_id = model_id

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }

        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        print(f"Loading model: {model_id}  (device={device})")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        print("Model ready.")

    def build_prompt(self, query: str, context: str = "") -> str:
        if context.strip():
            system = _DEFAULT_SYSTEM + "\n\nمعلومات من قاعدة المعرفة التونسية:\n" + context
        else:
            system = _DEFAULT_SYSTEM
        return _PROMPT_TEMPLATE.format(system=system, user=query)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_ids = self.model.generate(**inputs, generation_config=gen_config)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
