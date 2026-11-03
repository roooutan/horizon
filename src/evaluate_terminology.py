
"""Compare CS LoRA and Terminology LoRA on the same test data."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/kci_corpus/processed/final/sft_prepared"
SMOKE = ROOT / "experiments/terminology_eval/eval_smoke.jsonl"
CS_ADAPTER = ROOT / "models/finetuning_Bomin/outputs/hymt2_cs/adapter"
TERM_ADAPTER = ROOT / "models/terminology_full/adapter"
RESULTS = ROOT / "experiments/terminology_eval"
MODEL = "tencent/Hy-MT2-7B"


def load_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open(encoding="utf-8-sig") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def sample_key(row):
    payload = [
        row["id"], row["source"],
        row["target"], row["instruction"]
    ]
    raw = json.dumps(payload, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def boundary_contains(text, term):
    pattern = (
        r"(?<![A-Za-z0-9_])"
        + re.escape(term)
        + r"(?![A-Za-z0-9_])"
    )
    return bool(re.search(pattern, text, flags=re.I))


def terminology_counts(rows, name):
    correct = {"KEEP": 0, "TRANSLATE": 0}
    total = {"KEEP": 0, "TRANSLATE": 0}

    for row in rows:
        pred = row["predictions"][name]

        for hit in row.get("terms", []):
            if not hit.get("auto_enforce", True):
                continue

            action = hit.get("action")
            if action not in total:
                continue

            original = (
                hit.get("matched_text")
                or hit.get("term")
            )
            if not original:
                continue

            if action == "KEEP":
                ok = boundary_contains(pred, original)

            else:
                candidates = (
                    [hit.get("translation")]
                    + (hit.get("translation_variants") or [])
                )
                candidates = [
                    c for c in candidates
                    if isinstance(c, str) and c.strip()
                ]
                if not candidates:
                    continue

                ok = (
                    any(c in pred for c in candidates)
                    and not boundary_contains(pred, original)
                )

            total[action] += 1
            correct[action] += int(ok)

    return {
        action: {
            "correct": correct[action],
            "total": total[action],
            "rate": (
                round(correct[action] / total[action], 4)
                if total[action] else None
            )
        }
        for action in total
    }


def report(rows, selected_count, output_path):
    from sacrebleu.metrics import BLEU, CHRF

    paired = [
        r for r in rows
        if r.get("status") == "ok"
        and all(
            isinstance(r.get("predictions", {}).get(k), str)
            for k in ("cs", "term")
        )
    ]

    print(f"Compared pairs: {len(paired)}/{selected_count}")
    if not paired:
        return

    refs = [r["target"] for r in paired]

    metrics = {
        "evaluated_pairs": len(paired),
        "selected": selected_count,
        "models": {}
    }

    for name in ("cs", "term"):
        preds = [
            r["predictions"][name]
            for r in paired
        ]

        bleu = BLEU(tokenize="ko-mecab").corpus_score(
            preds, [refs]
        ).score
        chrf = CHRF().corpus_score(
            preds, [refs]
        ).score

        terms = terminology_counts(paired, name)

        metrics["models"][name] = {
            "bleu_ko_mecab": round(bleu, 4),
            "chrf": round(chrf, 4),
            "terms": terms
        }

        print(
            f"{name}: BLEU={bleu:.2f}, "
            f"chrF={chrf:.2f}"
        )
        print("  KEEP:", terms["KEEP"])
        print("  TRANSLATE:", terms["TRANSLATE"])

    metric_path = output_path.with_suffix(
        ".metrics.json"
    )
    metric_path.write_text(
        json.dumps(
            metrics, ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )
    print("Metrics saved:", metric_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        choices=["smoke", "full"],
        default="smoke"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="0 means all samples"
    )
    parser.add_argument(
        "--check-only",
        action="store_true"
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024
    )

    args = parser.parse_args()

    if (
        args.limit < 0
        or args.max_input_tokens <= 0
        or args.max_new_tokens <= 0
    ):
        parser.error("Invalid limit or token settings")

    # 소규모 데이터가 없으면 원본 Test에서 선택
    path = (
        SMOKE
        if args.set == "smoke" and SMOKE.is_file()
        else DATA / "sft_test.jsonl"
    )

    rows = load_jsonl(path)
    selected = (
        rows[:args.limit]
        if args.limit else rows
    )

    if not selected:
        raise ValueError("Empty evaluation set")

    # 로컬 점검에서는 GPU와 Adapter가 필요 없음
    if not args.check_only:
        for adapter in (CS_ADAPTER, TERM_ADAPTER):
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors"
            ):
                if not (adapter / name).is_file():
                    raise FileNotFoundError(adapter / name)

    RESULTS.mkdir(
        parents=True, exist_ok=True
    )

    output = RESULTS / (
        f"{args.set}_predictions.jsonl"
    )

    previous = (
        load_jsonl(output)
        if output.is_file() else []
    )
    done = {r["key"]: r for r in previous}

    if len(done) != len(previous):
        raise ValueError(
            "Duplicate results found"
        )

    valid_keys = {
        sample_key(row) for row in rows
    }
    if any(k not in valid_keys for k in done):
        raise ValueError(
            "Existing results belong to a different dataset"
        )

    pending = [
        r for r in selected
        if sample_key(r) not in done
    ]

    print("Dataset:", path)
    print("Selected:", len(selected))
    print("Already saved:", len(selected) - len(pending))

    if args.check_only:
        print("CHECK PASS (no GPU used)")
        return

    if pending:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer
        )
        from peft import PeftModel

        if not torch.cuda.is_available():
            raise RuntimeError(
                "RunPod CUDA GPU is required"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            trust_remote_code=True
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        base = AutoModelForCausalLM.from_pretrained(
            MODEL,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )

        model = PeftModel.from_pretrained(
            base,
            str(CS_ADAPTER),
            adapter_name="cs",
            is_trainable=False
        )

        model.load_adapter(
            str(TERM_ADAPTER),
            adapter_name="term",
            is_trainable=False
        )
        model.eval()

        print("Both LoRA adapters loaded.")

        with output.open("a", encoding="utf-8") as out:
            for i, row in enumerate(pending, 1):
                key = sample_key(row)

                messages = [{
                    "role": "user",
                    "content": (
                        row["instruction"].strip()
                        + "\n\n"
                        + row["source"]
                    )
                }]

                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )

                prompt_len = inputs["input_ids"].shape[-1]

                record = {
                    "key": key,
                    "id": row["id"],
                    "source": row["source"],
                    "target": row["target"],
                    "terms": row.get("terms", []),
                    "predictions": {}
                }

                if prompt_len > args.max_input_tokens:
                    record["status"] = "input_too_long"

                else:
                    inputs = {
                        k: v.to(model.device)
                        for k, v in inputs.items()
                    }

                    source_len = len(tokenizer.encode(
                        row["source"],
                        add_special_tokens=False
                    ))

                    budget = min(
                        args.max_new_tokens,
                        max(
                            256,
                            math.ceil(source_len * 3.5 + 128)
                        )
                    )

                    for name in ("cs", "term"):
                        model.set_adapter(name)
                        model.eval()

                        with torch.inference_mode():
                            tokens = model.generate(
                                **inputs,
                                max_new_tokens=budget,
                                do_sample=False,
                                num_beams=1,
                                pad_token_id=tokenizer.pad_token_id
                            )

                        answer = tokens[0][prompt_len:]

                        record["predictions"][name] = (
                            tokenizer.decode(
                                answer,
                                skip_special_tokens=True
                            ).strip()
                        )

                        record.setdefault(
                            "generated_tokens", {}
                        )[name] = len(answer)

                        del tokens

                    record["status"] = "ok"

                out.write(
                    json.dumps(
                        record, ensure_ascii=False
                    ) + "\n"
                )
                out.flush()

                done[key] = record

                print(
                    f"[{i}/{len(pending)}] "
                    f"id={row['id']} "
                    f"status={record['status']}",
                    flush=True
                )

    evaluated = [
        done[sample_key(row)]
        for row in selected
        if sample_key(row) in done
    ]

    report(evaluated, len(selected), output)


if __name__ == "__main__":
    main()
