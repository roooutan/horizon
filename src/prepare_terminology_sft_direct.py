
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FINAL = DATA / "kci_corpus" / "processed" / "final"
OUT = FINAL / "sft_prepared"

BASE = (
    "Translate the English academic text into natural Korean. "
    "Preserve technical proper names and any Markdown formatting."
)


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(name, rows):
    with (OUT / name).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_detector(glossary):
    lookup = {}

    for entry in glossary:
        options = [(entry["term"], entry["action"], False)]
        options += [
            (x, entry["action"], False)
            for x in entry.get("variants", [])
        ]

        if entry.get("abbreviation"):
            options.append(
                (entry["abbreviation"], "KEEP", True)
            )

        for phrase, action, abbrev in options:
            if not phrase or re.search("[가-힣]", phrase):
                continue

            sensitive = abbrev or bool(
                re.fullmatch(r"[A-Z][A-Z0-9-]{1,9}", phrase)
            )

            lookup.setdefault(phrase.casefold(), []).append(
                (phrase, entry, action, sensitive)
            )

    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(
            re.escape(x)
            for x in sorted(
                lookup,
                key=lambda x: (-len(x), x)
            )
        )
        + r")(?![A-Za-z0-9_])",
        re.IGNORECASE
    )

    return pattern, lookup


def find_terms(source, detector):
    pattern, lookup = detector
    found, seen = [], set()

    for m in pattern.finditer(source):
        for original, entry, action, sensitive in lookup[
            m.group().casefold()
        ]:
            if sensitive and m.group() != original:
                continue

            key = (
                m.group().casefold(),
                action,
                entry["term"]
            )

            if key not in seen:
                seen.add(key)
                found.append({
                    "term": entry["term"],
                    "matched_text": m.group(),
                    "action": action,
                    "translation": (
                        entry.get("translation")
                        if action == "TRANSLATE"
                        else None
                    ),
                    "translation_variants": (
                        entry.get("translation_variants") or []
                    ),
                    "auto_enforce": entry.get(
                        "auto_enforce", True
                    ),
                })
            break

    return found


def make_row(row, kind, detector):
    hits = find_terms(row["source_en"], detector)
    hard = [x for x in hits if x["auto_enforce"]]

    lines, used = [], set()

    for hit in hard:
        term = hit["matched_text"]
        key = (term.casefold(), hit["action"])

        if key in used:
            continue

        used.add(key)

        if hit["action"] == "KEEP":
            lines.append(
                f"- Keep '{term}' exactly in English."
            )
        else:
            preferred = hit["translation"]

            options = list(dict.fromkeys(
                x for x in (
                    [preferred] + hit["translation_variants"]
                )
                if x
            ))

            if len(options) > 1:
                lines.append(
                    f"- Translate '{term}' using a suitable "
                    f"Korean term from: {', '.join(options)}."
                )
            elif options:
                lines.append(
                    f"- Translate '{term}' as '{options[0]}'."
                )

    instruction = BASE

    if lines:
        instruction += (
            "\nFollow these terminology rules:\n"
            + "\n".join(lines)
        )

    return {
        "id": row["id"],
        "document_id": row["paper_id"],
        "sample_type": kind,
        "instruction": instruction,
        "source": row["source_en"],
        "target": row["target_ko"],
        "terms": hard,
    }


def main():
    paths = [
        DATA / "찐막glossary_curated.json",
        DATA / "glossary_curated.json",
    ]

    glossary_path = next(
        (p for p in paths if p.is_file()),
        None
    )

    if glossary_path is None:
        raise FileNotFoundError(
            f"Glossary not found: {paths}"
        )

    glossary = json.loads(
        glossary_path.read_text(encoding="utf-8-sig")
    )
    detector = prepare_detector(glossary)

    ready = {
        r["id"] for r in read_jsonl(
            FINAL / "terminology_curated_ready.jsonl"
        )
    }

    review = {
        r["id"] for r in read_jsonl(
            FINAL / "terminology_curated_review.jsonl"
        )
    }

    if ready & review:
        raise ValueError(
            "Audit ready and review IDs overlap"
        )

    strict, context, general = [], [], []
    train = list(read_jsonl(FINAL / "train.jsonl"))

    for row in train:
        if row["id"] in review:
            continue

        hits = find_terms(
            row["source_en"], detector
        )

        enforced = any(
            h["auto_enforce"] for h in hits
        )

        if row["id"] in ready:
            if not enforced:
                raise ValueError(
                    f"Audit ready mismatch: {row['id']}"
                )

            strict.append(
                make_row(
                    row, "terminology_rule", detector
                )
            )

        elif enforced:
            raise ValueError(
                f"Audit enforceable mismatch: {row['id']}"
            )

        elif hits:
            context.append(
                make_row(
                    row, "context_sensitive", detector
                )
            )

        else:
            general.append(
                make_row(
                    row, "general_translation", detector
                )
            )

    if (
        len(strict) != len(ready)
        or len(strict) + len(context)
        + len(general) + len(review) != len(train)
    ):
        raise ValueError(
            "Audit counts do not match train"
        )

    rng = random.Random(42)

    sampled_general = rng.sample(
        general, min(500, len(general))
    )

    train_rows = (
        strict + context + sampled_general
    )
    rng.shuffle(train_rows)

    valid_rows = [
        make_row(x, "evaluation", detector)
        for x in read_jsonl(FINAL / "valid.jsonl")
    ]

    test_rows = [
        make_row(x, "evaluation", detector)
        for x in read_jsonl(FINAL / "test.jsonl")
    ]

    groups = [
        train_rows, valid_rows, test_rows
    ]

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for field in ("id", "document_id"):
            first = {
                x[field] for x in groups[i]
            }
            second = {
                x[field] for x in groups[j]
            }

            if first & second:
                raise ValueError(
                    f"Split overlap: {field} {i}-{j}"
                )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    for name, rows in [
        ("sft_train.jsonl", train_rows),
        ("sft_valid.jsonl", valid_rows),
        ("sft_test.jsonl", test_rows),
    ]:
        save_jsonl(name, rows)

    report = {
        "glossary": glossary_path.name,
        "glossary_entries": len(glossary),
        "train": len(train_rows),
        "rules": len(strict),
        "context": len(context),
        "general": len(sampled_general),
        "review_excluded": len(review),
        "valid": len(valid_rows),
        "test": len(test_rows),
    }

    (OUT / "sft_preprocess_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("===== TERMINOLOGY SFT PREPROCESS =====")
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )
    print(f"Saved to: {OUT}")


if __name__ == "__main__":
    main()
