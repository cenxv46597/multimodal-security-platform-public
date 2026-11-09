"""Re-screen SARChat DeepSeek using the shared natural-language protocol."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from zipfile import ZipFile

from run_vlm_clean_screen import ROOT, load_json, sha256, timed
from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel

SNAPSHOT = ROOT / 'models/SARChat-DeepSeekVL-1.3B/master-20261108'
INVENTORY = ROOT / 'resources/manifests/sarchat_deepseek_inventory_20261108.json'
SOURCE = ROOT / '.work/deepseek-vl-source-681bffb'
ARCHIVE = ROOT / '.work/deepseek-vl-source-681bffb.zip'
DEPS = ROOT / '.work/deepseek-vl-deps'
PROTOCOL = ROOT / 'experiments/protocols/sarchat_glm_edge_natural_screen_20261109.json'
PLAN = 'bc712ce9733fa633e3a6628cea93da0f3777291dd36f011a98ac0152c117f4f9'
sys.path[:0] = [str(SOURCE), str(DEPS)]

def main():
    import torch
    from PIL import Image
    from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
    from transformers import AutoModelForCausalLM

    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=['probe', 'screen'])
    p.add_argument('--probe-report', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    output = a.output.resolve()
    if not output.is_relative_to(ROOT / 'runs') or output.exists():
        p.error('Choose a new output path under runs')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status':'running', 'phase':a.phase, 'runner_sha256':sha256(Path(__file__)),
              'inventory_sha256':sha256(INVENTORY), 'source_archive_sha256':sha256(ARCHIVE),
              'source_commit':'681bffb4519856ad27cc17531aacde31ddf6f1a7', 'protocol_sha256':sha256(PROTOCOL), 'predictions':[]}
    start = time.perf_counter()
    try:
        inventory = load_json(INVENTORY)
        protocol = load_json(PROTOCOL)
        if protocol['grouped_plan_sha256'] != PLAN: raise ValueError('Panel mismatch')
        for e in inventory['files']:
            path = (SNAPSHOT / e['path']).resolve()
            if not path.is_relative_to(SNAPSHOT) or path.stat().st_size != e['size'] or sha256(path) != e['sha256']:
                raise ValueError('Snapshot verification failed: ' + e['path'])
        with ZipFile(ARCHIVE) as z:
            for name in z.namelist():
                parts = Path(name).parts
                if len(parts) > 2 and parts[1] == 'deepseek_vl' and name.endswith('.py'):
                    if hashlib.sha256(z.read(name)).hexdigest() != sha256(SOURCE.joinpath(*parts[1:])):
                        raise ValueError('Source code mismatch')
        _, _, _, rows, _, _, labels, screen = load_panel(PLAN)
        if a.phase == 'screen':
            if a.probe_report is None: raise ValueError('Screen requires probe')
            probe=load_json(a.probe_report)
            if probe['status']!='passed' or any(probe[k]!=report[k] for k in ('runner_sha256','inventory_sha256','source_archive_sha256','protocol_sha256')):
                raise ValueError('Probe mismatch')
            report['probe_sha256']=sha256(a.probe_report)
        torch.set_num_threads(4)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        processor = VLChatProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
        model, info = AutoModelForCausalLM.from_pretrained(SNAPSHOT, local_files_only=True,
            torch_dtype=torch.float16, attn_implementation='eager', output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Checkpoint loading issue: '+str(info))
        if not isinstance(model, MultiModalityCausalLM): raise ValueError('Unexpected model')
        model.eval().requires_grad_(False).to('cuda')
        report.update(model=inventory['model'], grouped_plan_sha256=PLAN, dtype='float16',
                      gpu=torch.cuda.get_device_name(), prompt=protocol['question'],
                      generation=protocol['generation'], image_tokens=processor.num_image_tokens)
        def infer(row):
            conversation=[{'role':'User','content':'<image_placeholder>\n'+protocol['question']},
                          {'role':'Assistant','content':''}]
            with Image.open(ROOT / row['asset']['file']) as source:
                im=source.convert('RGB')
                inputs=processor(conversations=conversation, images=[im], force_batchify=True).to('cuda',dtype=torch.float16)
            if int(inputs.images_seq_mask.sum()) != 576 or not torch.isfinite(inputs.pixel_values).all():
                raise ValueError('Unexpected image input')
            with torch.inference_mode():
                embeds=model.prepare_inputs_embeds(**inputs)
                if not bool(torch.isfinite(embeds).all()): raise ValueError('Nonfinite embeddings')
                out=model.language_model.generate(inputs_embeds=embeds, attention_mask=inputs.attention_mask,
                    pad_token_id=processor.tokenizer.eos_token_id, bos_token_id=processor.tokenizer.bos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id, **protocol['generation'],
                    use_cache=True, return_dict_in_generate=True, output_logits=True)
                if not out.logits or not all(bool(torch.isfinite(x).all()) for x in out.logits):
                    raise ValueError('Nonfinite or missing generation logits')
            answer=processor.tokenizer.decode(out.sequences[0].cpu().tolist(),skip_special_tokens=True)
            return {'answer':answer,'generation_logits_finite':True,'generated_tokens':int(out.sequences.shape[1]), 'hit_token_limit':int(out.sequences.shape[1]) >= protocol['generation']['max_new_tokens']}
        if a.phase=='probe':
            timed(report,'probe',lambda:infer(rows[0]),'cuda')
            report.update(status='passed',probe_sample_id=rows[0]['sample_id'])
        else:
            for row in rows:
                item={'sample_id':row['sample_id'],'status':'ok'}
                try:
                    timed(item,'generation',lambda:infer(row),'cuda')
                    item.update(item.pop('generation'))
                except Exception as e:
                    item.update(status='error',answer=None,error_type=type(e).__name__,error=str(e))
                report['predictions'].append(item)
                output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
                print('Evaluated',len(report['predictions']),'/60',flush=True)
            report['score']=score_samples(rows,report['predictions'],labels,screen)
            report['status']='completed'
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e))
        traceback.print_exc()
    report['seconds']=time.perf_counter()-start
    if torch.cuda.is_available(): report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(report['status'],output,flush=True)
    return int(report['status']=='failed')

if __name__=='__main__':
    raise SystemExit(main())
