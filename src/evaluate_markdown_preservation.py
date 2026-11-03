#!/usr/bin/env python3
"""Controlled Markdown structural preservation pilot for the same CS and Terminology LoRAs.

36 synthetic English Markdown documents, paired same-prompt evaluation.
Scores syntax/structure only, not translation meaning. Automatically identified
structure features may produce false positives/negatives; manually audit examples.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path('/workspace')
DEFAULT_DATA = ROOT / 'markdown_eval_cases.jsonl'
DEFAULT_OUT = ROOT / 'experiments/markdown_eval'
CS_PATH = ROOT / 'models/finetuning_Bomin/outputs/hymt2_cs/adapter'
TERM_PATH = ROOT / 'models/terminology_full/adapter'
BASE_MODEL = 'tencent/Hy-MT2-7B'
EXPECTED_CATS = {'headings', 'lists', 'tables', 'code', 'math_inline', 'references'}


def read_jsonl(path):
    with open(path, encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'{path}, line {line_no}: {exc}') from exc


def fingerprint(row):
    data = [row['id'], row['source_md'], row['instruction'], row['category']]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode('utf-8')).hexdigest()


def extract_features(md):
    """Extract Markdown structural fingerprints, *not* Korean translation accuracy."""
    md = md.replace('\r\n', '\n')
    lines = md.splitlines()
    features = {}
    headings = re.findall(r'^\s{0,3}(#{1,6})\s+\S.*$', md, re.M)
    if headings:
        features['headings'] = tuple(len(x) for x in headings)

    # Fenced blocks: language tags and exact body (ignoring trailing newline only).
    fences = re.findall(r'^\s{0,3}```([^\n`]*)\n(.*?)^\s{0,3}```\s*$', md, re.M | re.S)
    if fences:
        features['fenced_code'] = tuple((lang.strip(), body.rstrip('\n')) for lang, body in fences)
    no_fences = re.sub(r'^\s{0,3}```[^\n`]*\n.*?^\s{0,3}```\s*$', '', md, flags=re.M | re.S)

    # List item nesting (indent rank) and list kind, including checklist state.
    list_items = []
    list_regex = re.compile(r'^( *)([-*+]|\d+[.)])\s+(\[[ xX]\]\s*)?\S.*$')
    for line in lines:
        match = list_regex.match(line)
        if match:
            indent, marker, check = match.groups()
            kind = 'ordered' if marker[0].isdigit() else 'unordered'
            if check is not None:
                kind = 'checkbox_checked' if 'x' in check.lower() else 'checkbox_unchecked'
            list_items.append((len(indent), kind))
    if list_items:
        indentation_levels = {x: i for i, x in enumerate(sorted({v for v, _ in list_items}))}
        features['list_structure'] = tuple((indentation_levels[n], kind) for n, kind in list_items)

    # Pipe tables. All benchmark tables deliberately use outer pipes.
    pipe_lines = [line.strip() for line in lines if line.lstrip().startswith('|') and line.rstrip().endswith('|')]
    if pipe_lines:
        cells = [len(re.split(r'(?<!\\)\|', line.strip('|'))) for line in pipe_lines]
        separators = [bool(re.fullmatch(r'\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*', line)) for line in pipe_lines]
        features['table_shape'] = (tuple(cells), tuple(separators))

    inline = re.findall(r'(?<!`)`([^`\n]+)`(?!`)', no_fences)
    if inline:
        features['inline_code'] = tuple(inline)

    block_math = re.findall(r'(?<!\$)\$\$(.*?)\$\$(?!\$)', no_fences, flags=re.S)
    without_block = re.sub(r'(?<!\$)\$\$.*?\$\$(?!\$)', '', no_fences, flags=re.S)
    inline_math = re.findall(r'(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)', without_block)
    if block_math or inline_math:
        normalize = lambda x: re.sub(r'\s+', '', x)
        features['math_expressions'] = (tuple(map(normalize, block_math)), tuple(map(normalize, inline_math)))

    urls = re.findall(r'\[[^\]\n]+\]\((https?://[^\s)]+)\)', no_fences)
    if urls:
        features['link_urls'] = tuple(urls)

    footnotes = re.findall(r'\[\^([A-Za-z0-9_-]+)\]', no_fences)
    if footnotes:
        features['footnote_ids'] = tuple(footnotes)

    quotes = re.findall(r'^\s{0,3}(>+)\s+.*$', no_fences, re.M)
    if quotes:
        features['blockquote_depths'] = tuple(map(len, quotes))

    bold = re.findall(r'(?<!\*)\*\*([^\n*]+)\*\*(?!\*)', no_fences)
    if bold:
        features['bold_spans'] = len(bold)
    italic = re.findall(r'(?<![\w*])\*([^\n*]+)\*(?!\*)', no_fences)
    if italic:
        features['italic_spans'] = len(italic)

    return features


def judge_structure(source, prediction):
    expected = extract_features(source)
    found = extract_features(prediction)
    comparison = {k: bool(found.get(k) == v) for k, v in expected.items()}
    return expected, found, comparison


def analyze(rows, output_dir, selected):
    report_rows = []
    for row in rows:
        if row.get('status') != 'ok':
            continue
        if any(row.get('hit_cap', {}).get(model, False) for model in ('cs', 'term')):
            continue
        if any(not isinstance(row.get('predictions', {}).get(model), str) for model in ('cs', 'term')):
            continue
        for name in ('cs', 'term'):
            _, actual, matches = judge_structure(row['source_md'], row['predictions'][name])
            for feature, correct in matches.items():
                report_rows.append({'id': row['id'], 'category': row['category'], 'model': name,
                                    'feature': feature, 'correct': int(correct),
                                    'source_md': row['source_md'], 'output_md': row['predictions'][name],
                                    'actual_feature': str(actual.get(feature, 'NOT_FOUND'))[:400]})
    by_model = {}
    all_features = sorted({r['feature'] for r in report_rows})
    for model in ('cs', 'term'):
        unit = [r for r in report_rows if r['model'] == model]
        by_model[model] = {'feature_checks': len(unit), 'correct_checks': sum(r['correct'] for r in unit),
                           'feature_match_rate': round(sum(r['correct'] for r in unit) / len(unit), 4) if unit else None,
                           'by_feature': {}}
        for feature in all_features:
            sub = [r for r in unit if r['feature'] == feature]
            by_model[model]['by_feature'][feature] = {
                'n': len(sub), 'correct': sum(r['correct'] for r in sub),
                'rate': round(sum(r['correct'] for r in sub)/len(sub), 4) if sub else None}
    num_pairs = len({r['id'] for r in report_rows})
    summary = {'benchmark': 'synthetic_markdown_preservation_pilot',
               'description': 'Fixed 36 synthetic Markdown documents; identical explicit Markdown-preservation prompts for both models.',
               'warning': 'Structural-marker proxy, not translation quality nor complete Markdown parser validation. Audit failures manually.',
               'selected': selected, 'paired_scorable': num_pairs, 'scored_opportunities_per_model': len([r for r in report_rows if r['model']=='cs']),
               'models': by_model}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'markdown_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output_dir/'markdown_feature_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['id','category','model','feature','correct','source_md','output_md','actual_feature'])
        writer.writeheader();writer.writerows(report_rows)
    print(f'Paired, scorable cases: {num_pairs} / {selected}',flush=True)
    for name in ('cs','term'):
        m=by_model[name]
        print(f'{name}: {m["correct_checks"]}/{m["feature_checks"]} feature checks preserved | rate={m["feature_match_rate"]}',flush=True)
        for key, sub in m['by_feature'].items():
            print(f'  {key}: {sub["correct"]}/{sub["n"]}',flush=True)
    print(f'Results: {output_dir}/markdown_metrics.json | {output_dir}/markdown_feature_audit.csv',flush=True)


def load_saved(path):
    seen={}
    if not path.exists():
        return seen
    with path.open('rb+') as f:
        while True:
            start=f.tell()
            b=f.readline()
            if not b: break
            if not b.strip(): continue
            try:
                rec=json.loads(b.decode('utf8'))
                key=rec['fingerprint']
                if key in seen: raise ValueError('duplicate record')
                seen[key]=rec
            except Exception as exc:
                # Recover only an incomplete final line.
                rest=f.read(1)
                if rest:
                    raise ValueError(f'Corrupted middle record at byte {start}') from exc
                f.truncate(start)
                print(f'Removed incomplete trailing result at byte {start}',flush=True)
                break
    return seen


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--limit', type=int, default=0, help='0=all 36 synthetic documents; 6=smoke test')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=512)
    args=p.parse_args()
    if args.limit<0 or args.max_new_tokens<64: p.error('invalid limit or tokens')
    cases=list(read_jsonl(args.data))
    if len(cases)!=36 or collections.Counter(r['category'] for r in cases)!=collections.Counter({c:6 for c in EXPECTED_CATS}):
        raise ValueError('Expected 36 Markdown cases, six per category')
    if len({fingerprint(x) for x in cases})!=36: raise ValueError('Duplicate input cases')
    for r in cases:
        if not extract_features(r['source_md']):raise ValueError('Unscorable Markdown case '+r['id'])
    selected=cases[:args.limit] if args.limit else cases
    print(f'DATA CHECK: PASS | 36 cases | selected={len(selected)}',flush=True)
    if args.check_only:
        hist=collections.Counter(k for r in cases for k in extract_features(r['source_md']))
        print('Feature counts:',dict(sorted(hist.items())),flush=True)
        print('CHECK PASS (no GPU used)',flush=True)
        return
    for label,path in [('CS',CS_PATH),('TERM',TERM_PATH)]:
        for file in ('adapter_config.json','adapter_model.safetensors'):
            if not (path/file).is_file():raise FileNotFoundError(f'{label}: {path/file}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/'markdown_predictions.jsonl'
    saved=load_saved(output)
    all_keys={fingerprint(r) for r in cases}
    if any(k not in all_keys for k in saved):raise ValueError('Output contains results for a different dataset')
    pending=[r for r in selected if fingerprint(r) not in saved]
    print(f'Already saved: {len(selected)-len(pending)} | Remaining: {len(pending)}',flush=True)
    if pending:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
        tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base=AutoModelForCausalLM.from_pretrained(BASE_MODEL,dtype=dtype,device_map='auto',trust_remote_code=True)
        model=PeftModel.from_pretrained(base,str(CS_PATH),adapter_name='cs',is_trainable=False)
        model.load_adapter(str(TERM_PATH),adapter_name='term',is_trainable=False)
        model.eval();model.config.use_cache=True
        print('Both adapters loaded; Markdown generation started.',flush=True)
        eosval=model.generation_config.eos_token_id
        eos=set(eosval) if isinstance(eosval,(list,tuple)) else ({eosval} if eosval is not None else set())
        if tokenizer.eos_token_id is not None:eos.add(tokenizer.eos_token_id)
        with output.open('a',encoding='utf8') as f:
            for n,row in enumerate(pending,1):
                prompt=row['instruction']+'\n\n'+row['source_md']
                encoded=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
                promptlen=encoded['input_ids'].shape[-1]
                inputs={k:v.to(model.device) for k,v in encoded.items()}
                rec={'fingerprint':fingerprint(row),'id':row['id'],'category':row['category'],
                     'source_md':row['source_md'], 'instruction':row['instruction'],
                     'predictions':{},'hit_cap':{},'status':'ok'}
                for name in ('cs','term'):
                    model.set_adapter(name);model.eval()
                    with torch.inference_mode():
                        gen=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=1,
                                           use_cache=True,pad_token_id=tokenizer.pad_token_id)
                    ids=gen[0][promptlen:].tolist()
                    rec['predictions'][name]=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                    rec['hit_cap'][name]=len(ids)>=args.max_new_tokens and not any(tok in eos for tok in ids)
                    del gen
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                saved[rec['fingerprint']]=rec
                print(f'[{n}/{len(pending)}] {row["id"]} | cs_cap={rec["hit_cap"]["cs"]} term_cap={rec["hit_cap"]["term"]}',flush=True)
    analyze([saved[fingerprint(r)] for r in selected],args.output_dir,len(selected))


if __name__=='__main__':main()
