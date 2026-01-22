# convert_math500.py
import json
from datasets import load_dataset

def main():
    # 不同仓库/版本字段名可能略不同：先 print(ds.column_names) 看一下
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")  # 例子：按你实际的dataset名改

    out = []
    for ex in ds:
        # 常见字段候选：problem / question / prompt
        prompt = ex.get("problem")
        # 常见答案字段候选：answer / final_answer / solution
        gt = ex.get("answer")

        if prompt is None or gt is None:
            raise KeyError(f"字段没对上：{ex.keys()}")

        out.append({
            "user_prompt": prompt,
            "ground_truth": str(gt).strip(),
        })

    with open("math-500.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("done: math-500.json / math-500.jsonl")

if __name__ == "__main__":
    main()
