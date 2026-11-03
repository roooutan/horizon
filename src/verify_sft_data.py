"""Final model-agnostic SFT data checks. Run: python src/verify_sft_data.py"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'kci_corpus' / 'processed' / 'final' / 'sft_prepared'
FILES = {'train': 'sft_train.jsonl', 'valid': 'sft_valid.jsonl', 'test': 'sft_test.jsonl'}
REQUIRED = ('id', 'document_id', 'instruction', 'source', 'target', 'terms')


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Run prepare_terminology_sft_fixed_v2.py first.')
    with path.open(encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'Invalid JSON: {path.name}:{line_no}') from e
                if not isinstance(obj, dict) or any(k not in obj for k in REQUIRED):
                    raise ValueError(f'Missing required fields: {path.name}:{line_no}')
                for key in ('id', 'document_id', 'instruction', 'source', 'target'):
                    if not isinstance(obj[key], str) or not obj[key].strip():
                        raise ValueError(f'Empty/invalid {key}: {path.name}:{line_no}')
                if not isinstance(obj['terms'], list):
                    raise ValueError(f'Invalid terms: {path.name}:{line_no}')
                if any(not x.get('auto_enforce', False) for x in obj['terms']):
                    raise ValueError(f'Non-enforced term in strict list: {path.name}:{line_no}')
                yield obj


def norm(text):
    return ' '.join(text.split()).casefold()


def main():
    groups = {split: list(read_jsonl(DATA / name)) for split, name in FILES.items()}
    counts = {name: len(rows) for name, rows in groups.items()}
    assert counts == {'train': 1930, 'valid': 620, 'test': 624}, f'Unexpected split counts: {counts}'
    identifiers = {s: {r['id'] for r in a} for s, a in groups.items()}
    documents = {s: {r['document_id'] for r in a} for s, a in groups.items()}
    source_texts = {s: {norm(r['source']) for r in a} for s, a in groups.items()}
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        assert not (identifiers[a] & identifiers[b]), f'Overlapping IDs: {a}/{b}'
        assert not (documents[a] & documents[b]), f'Overlapping document IDs: {a}/{b}'
        assert not (source_texts[a] & source_texts[b]), f'Overlapping source texts: {a}/{b}'
    kinds = Counter(r.get('sample_type', '') for r in groups['train'])
    assert kinds == {'terminology_rule': 243, 'context_sensitive': 1187, 'general_translation': 500}, kinds
    max_chars = max(len(r['instruction']) + len(r['source']) + len(r['target']) for rows in groups.values() for r in rows)
    print('===== SFT DATA PRE-FLIGHT =====')
    print('Train / Valid / Test:', counts['train'], '/', counts['valid'], '/', counts['test'])
    print('Train sample types:', dict(kinds))
    print('Required fields / nonempty data: PASS')
    print('Cross-split ID / paper / English source overlap: 0')
    print('Longest instruction + source + target (characters):', max_chars)
    print('RESULT: PASS (data structure; not a guarantee of translation quality)')
    print('NEXT: select model/checkpoint, tokenizer, GPU VRAM, and train LoRA/QLoRA')


if __name__ == '__main__':
    main()
