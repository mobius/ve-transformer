"""Generate a scoped VE model factory from pinned upstream; never edit upstream."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'build/vendor/llama.cpp/src/llama-model.cpp'
text=source.read_text()
start=text.index('static llama_model * llama_model_mapping(')
end=text.index('\nllama_model * llama_model_create(',start)
factory='''// VE target intentionally supports the selected Qwen3.5/3.6 MoE architecture.
static llama_model * llama_model_mapping(llm_arch arch, const llama_model_params & params) {
    if (arch == LLM_ARCH_QWEN35MOE) return new llama_model_qwen35moe(params);
    throw std::runtime_error("NEC VE build supports only qwen35moe architecture");
}
'''
target=ROOT/'build/llama-ve/qwen-ve-model.cpp'
target.parent.mkdir(parents=True,exist_ok=True)
new=text[:start]+factory+text[end:]
if not target.exists() or target.read_text()!=new:
    target.write_text(new)
print('Generated VE factory for actual qwen35moe target; upstream is unchanged.')
