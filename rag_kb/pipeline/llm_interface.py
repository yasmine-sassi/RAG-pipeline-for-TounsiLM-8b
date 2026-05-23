import os
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Reduces memory fragmentation on the CUDA allocator — helps when the KV cache
# grows and freed blocks don't line up with new allocation sizes.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "alabenayed/TounsiLM-8b"

_PROMPT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n"
    "{system}\n"
    "<</SYS>>\n\n"
    "{user} [/INST]"
)

_DEFAULT_SYSTEM = (
    "أنت مساعد ذكي متخصص في اللهجة التونسية والثقافة التونسية. "
    "أجب على سؤال المستخدم مباشرةً بأسلوب طبيعي وواضح. "
    "إذا أُعطيت ملاحظات مرجعية، استخدمها فقط إذا كانت تتعلق مباشرةً بسؤال المستخدم. "
    "إذا لم تكن الملاحظات ذات صلة، تجاهلها تماماً وأجب من معرفتك العامة. "
    "لا تسرد المصادر ولا تكرر محتوى الملاحظات — أجب بكلامك الخاص فقط."
)


def _load_model(model_id: str, device: str):
    kwargs: dict = {"trust_remote_code": True}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


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

        print(f"Loading model: {model_id}  (device={device})")
        self.model = _load_model(model_id, device)

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()

        # Use the actual device of the embedding layer, not self.device.
        # With device_map="auto" some layers may land on CPU even when CUDA is available.
        self._input_device = next(self.model.parameters()).device
        print(f"Model ready.  (input device: {self._input_device})")

    def build_prompt(self, query: str, context: str = "") -> str:
        if context.strip():
            system = (
                _DEFAULT_SYSTEM
                + "\n\n[ملاحظات مرجعية — استخدمها لتعزيز إجابتك، لا لنسخها]\n"
                + context
            )
        else:
            system = _DEFAULT_SYSTEM
        return _PROMPT_TEMPLATE.format(system=system, user=query)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        if self._input_device.type == "cuda":
            torch.cuda.empty_cache()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,   # hard cap to limit KV cache memory
        ).to(self._input_device)

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
