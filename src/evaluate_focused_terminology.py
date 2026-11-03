#!/usr/bin/env python3
"""Controlled synthetic terminology adherence test (CS vs terminology LoRA).

No reference translations used in generation. The glossary condition explicitly
provides the desired term policy to BOTH models. Results are NOT KCI BLEU/chrF.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/workspace')
DEFAULT_DATA = ROOT / 'terminology_focused_test.jsonl'
DEFAULT_OUT = ROOT / 'experiments/terminology_focused'
CS_PATH = ROOT / 'models/finetuning_Bomin/outputs/hymt2_cs/adapter'
TERM_PATH = ROOT / 'models/terminology_full/adapter'
BASE_MODEL = 'tencent/Hy-MT2-7B'


def read_jsonl(path):
    with path.open('r', encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as exc:
                    raise ValueError(f'Invalid JSONL: {path}, line {n}: {exc}') from exc


def fingerprint(row):
    core = [row['id'], row['source'], row['instruction'], row['term'], row['action'], row['condition']]
    return hashlib.sha256(json.dumps(core, ensure_ascii=False).encode('utf-8')).hexdigest()


def normalize_korean(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def has_english_phrase(text, phrase):
    # Whole-word search for short acronyms, hyphens, technical model names.
    return bool(re.search(r'(?<![A-Za-z0-9_])' + re.escape(phrase) + r'(?![A-Za-z0-9_])',
                          text, flags=re.IGNORECASE))


def judge(row, answer):
    if row['action'] == 'KEEP':
        accepted = row['accepted_keep']
        return any(has_english_phrase(answer, t) for t in accepted), None
    if row['action'] == 'TRANSLATE':
        normalized = normalize_korean(answer)
        accepted = row['accepted_translations']
        passed = any(normalize_korean(t) in normalized for t in accepted if t)
        original_in_output = has_english_phrase(answer, row['term'])
        return passed, original_in_output
    raise ValueError('Unexpected policy: '+str(row['action']))


def score(rows, output_dir):
    usable = [r for r in rows if r.get('status') == 'ok'
              and not any(r.get('hit_cap', {}).get(m, False) for m in ('cs', 'term'))
              and all(isinstance(r.get('predictions', {}).get(m), str) for m in ('cs', 'term'))]
    summary = {'method': 'controlled_synthetic_glossary_adherence',
               'important': 'One synthetic context per glossary term; not translation quality or BLEU; manually audit edge cases.',
               'total_saved':len(rows),'paired_scorable':len(usable),
               'by_condition':{}, 'unscored':len(rows)-len(usable)}
    audit=[]
    for r in usable:
        cs, cs_original = judge(r, r['predictions']['cs'])
        term, term_original = judge(r, r['predictions']['term'])
        audit.append({'id':r['id'], 'condition':r['condition'],'action':r['action'],
                      'term':r['term'], 'expected':r['expected'], 'source':r['source'],
                      'cs_correct':int(cs),'term_correct':int(term),
                      'cs_source_phrase_retained':cs_original,'term_source_phrase_retained':term_original,
                      'cs_output':r['predictions']['cs'],'term_output':r['predictions']['term']})
    for condition in ('plain','glossary'):
        by_condition = {}
        for action in ('KEEP','TRANSLATE','ALL'):
            arr=[x for x in audit if x['condition']==condition and (action=='ALL' or x['action']==action)]
            n=len(arr); c=sum(x['cs_correct'] for x in arr); t=sum(x['term_correct'] for x in arr)
            by_condition[action] = {'n':n,
                'cs_correct':c,'cs_rate':round(c/n,4) if n else None,
                'term_correct':t,'term_rate':round(t/n,4) if n else None,
                'term_minus_cs_percentage_points':round(100*(t-c)/n,2) if n else None,
                'term_only_correct':sum(x['term_correct']==1 and x['cs_correct']==0 for x in arr),
                'cs_only_correct':sum(x['cs_correct']==1 and x['term_correct']==0 for x in arr),
                'both_correct':sum(x['cs_correct']==1 and x['term_correct']==1 for x in arr),
                'both_wrong':sum(x['cs_correct']==0 and x['term_correct']==0 for x in arr)}
        summary['by_condition'][condition]=by_condition
    output_dir.mkdir(parents=True,exist_ok=True)
    metrics_file=output_dir/'focused_metrics.json'
    metrics_file.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    audit_file=output_dir/'focused_audit.csv'
    columns=['id','condition','action','term','expected','source','cs_correct','term_correct',
             'cs_source_phrase_retained','term_source_phrase_retained','cs_output','term_output']
    with audit_file.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(audit)
    print('\n======= FOCUSED TERMINOLOGY RESULTS =======',flush=True)
    print(f'Paired, scorable cases: {len(usable)} / {len(rows)}')
    for condition in ('plain','glossary'):
        print(f'--- {condition} ---')
        for action in ('KEEP','TRANSLATE','ALL'):
            m=summary['by_condition'][condition][action]
            print(f'  {action}: {m["n"]} | CS {m["cs_correct"]}/{m["n"]} | TERM {m["term_correct"]}/{m["n"]} | delta {m["term_minus_cs_percentage_points"]} pp | TERM-only {m["term_only_correct"]} | CS-only {m["cs_only_correct"]}')
    print(f'METRICS: {metrics_file}\nAUDIT: {audit_file}',flush=True)


def load_previous(path):
    """Recover only an incomplete trailing record; reject a corrupted middle record."""
    saved={}
    if not path.exists(): return saved
    with path.open('rb+') as f:
        while True:
            start=f.tell(); b=f.readline()
            if not b: break
            if not b.strip(): continue
            try:
                r=json.loads(b.decode('utf-8'))
                assert isinstance(r['fingerprint'],str)
                assert isinstance(r['predictions'],dict)
                if r['fingerprint'] in saved: raise ValueError('duplicate fingerprint')
                saved[r['fingerprint']]=r
            except Exception as ex:
                if f.read(1):
                    raise ValueError(f'Corrupted middle record at byte {start}') from ex
                f.truncate(start)
                print(f'Recovered interrupted trailing JSONL line at byte {start}',flush=True)
                break
    return saved


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    p.add_argument('--limit',type=int,default=0,help='0=all (190 cases); 10=smoke test')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=192)
    args=p.parse_args()
    if args.limit<0 or args.max_new_tokens<32:p.error('Invalid limit/tokens')
    data=list(read_jsonl(args.data))
    if len(data)!=190: raise ValueError(f'Expected 190 controlled cases, got {len(data)}')
    if len({fingerprint(x) for x in data})!=len(data): raise ValueError('Duplicate cases')
    if Counter(x['action'] for x in data)!=Counter({'KEEP':52,'TRANSLATE':138}): raise ValueError('Unexpected policy counts')
    rows=data[:args.limit] if args.limit else data
    print('DATA CHECK: PASS | total cases=190 | selected='+str(len(rows)),flush=True)
    if args.check_only:
        print('CHECK PASS (no GPU used)',flush=True)
        return
    for label,path in [('CS',CS_PATH),('TERM',TERM_PATH)]:
        for n in ('adapter_config.json','adapter_model.safetensors'):
            if not (path/n).is_file():raise FileNotFoundError(f'{label} missing {path/n}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result_file=args.output_dir/'focused_predictions.jsonl'
    saved=load_previous(result_file)
    all_keys={fingerprint(x) for x in data}
    if any(k not in all_keys for k in saved):
        raise ValueError('Old output belongs to different data. Change --output-dir')
    pending=[x for x in rows if fingerprint(x) not in saved]
    print(f'Already saved: {len(rows)-len(pending)} | Remaining: {len(pending)}',flush=True)
    if pending:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
        tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base=AutoModelForCausalLM.from_pretrained(BASE_MODEL,dtype=dtype,device_map='auto',trust_remote_code=True)
        model=PeftModel.from_pretrained(base,str(CS_PATH),adapter_name='cs',is_trainable=False)
        model.load_adapter(str(TERM_PATH),adapter_name='term',is_trainable=False)
        model.eval();model.config.use_cache=True
        print('Both LoRA adapters loaded. Starting focused evaluation.',flush=True)
        eos_val=model.generation_config.eos_token_id
        eos=set(eos_val) if isinstance(eos_val,(list,tuple)) else ({eos_val} if eos_val is not None else set())
        if tokenizer.eos_token_id is not None:eos.add(tokenizer.eos_token_id)
        with result_file.open('a',encoding='utf-8') as out:
            for i,row in enumerate(pending,1):
                msg=[{'role':'user','content':row['instruction']+'\n\n'+row['source']}]
                enc=tokenizer.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
                promptlen=enc['input_ids'].shape[-1]
                inputs={k:v.to(model.device) for k,v in enc.items()}
                rec={k:row[k] for k in ('id','source','instruction','condition','action','term','expected','accepted_keep','accepted_translations')}
                rec.update({'fingerprint':fingerprint(row),'status':'ok','predictions':{},'hit_cap':{}})
                for label in ('cs','term'):
                    model.set_adapter(label);model.eval()
                    with torch.inference_mode():
                        generated=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=1,use_cache=True,pad_token_id=tokenizer.pad_token_id)
                    suffix=generated[0][promptlen:].tolist()
                    rec['predictions'][label]=tokenizer.decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
                    rec['hit_cap'][label]=len(suffix)>=args.max_new_tokens and not any(x in eos for x in suffix)
                    del generated
                out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
                saved[rec['fingerprint']]=rec
                print(f'[{i}/{len(pending)}] {row["condition"]} | {row["action"]} | {row["term"]} | cs_cap={rec["hit_cap"]["cs"]} term_cap={rec["hit_cap"]["term"]}',flush=True)
    selected=[saved[fingerprint(x)] for x in rows]
    score(selected,args.output_dir)

if __name__=='__main__': main()
