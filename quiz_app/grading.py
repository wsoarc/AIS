"""Behavioral checks on small, deterministic CPU fixtures; no notebook execution."""
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class LimitedOutput(io.TextIOBase):
    def __init__(self): self.text = ''
    def write(self, value):
        self.text += value[:max(0, 3000 - len(self.text))]
        return len(value)
    def flush(self): pass

def freeze(value):
    import numpy as np
    import torch
    from types import SimpleNamespace
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        return {'tensor':t.tolist(), 'shape':list(t.shape), 'dtype':str(t.dtype)}
    if isinstance(value, np.ndarray):
        return {'array':value.tolist(), 'shape':list(value.shape), 'dtype':str(value.dtype)}
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, SimpleNamespace): return freeze(vars(value))
    if isinstance(value, dict): return {k:freeze(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [freeze(v) for v in value]
    if value is None or isinstance(value,(str,bool,int,float)): return value
    raise TypeError('채점 결과로 변환할 수 없는 객체: ' + type(value).__name__)

def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-6)
    return type(a) is type(b) and a == b

def execute(q, code, case):
    import torch
    import numpy as np
    import asyncio
    import random
    from types import SimpleNamespace
    torch.set_num_threads(1)
    torch.manual_seed(173+case)
    np.random.seed(173+case)
    random.seed(173+case)
    namespace=dict(torch=torch,np=np,nn=torch.nn,F=torch.nn.functional,
                   math=math,NS=SimpleNamespace,asyncio=asyncio,case=case)
    capture=LimitedOutput()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        exec(compile('from __future__ import annotations\n'+q['setup'], '<provided>', 'exec'),namespace)
        exec(compile('from __future__ import annotations\n'+code,'<answer>','exec'),namespace)
        exec(compile(q['fixture'],'<fixture>','exec'),namespace)
        result=freeze(namespace['result'])
    return result,capture.text

def check_in_process(q, answer):
    from quiz_app.catalog import assemble
    checks=[]
    for case,label in enumerate(q['cases']):
        try:
            expected,_=execute(q,assemble(q,q['answer']),case)
        except Exception as exc:
            return dict(status='unavailable',message='채점 환경 또는 기준 코드 오류: '+type(exc).__name__+': '+str(exc)[:300])
        try:
            actual,output=execute(q,assemble(q,answer),case)
            passed=equal(actual,expected)
            checks.append(dict(label=label,passed=passed,message='통과' if passed else '출력값·구조·dtype 또는 호출 결과가 기대와 다릅니다.',
                               expected=expected,actual=actual,output=output))
        except Exception as exc:
            checks.append(dict(label=label,passed=False,message=type(exc).__name__+': '+str(exc)[:400]))
    fraction=sum(c['passed'] for c in checks)/len(checks)
    return dict(status='graded',fraction=fraction,score=round(q['points']*fraction,2),max_score=q['points'],checks=checks)

def grade(q, answer, timeout=20):
    # Separate process, temporary cwd, minimal environment and bounded lifetime.
    # This is for the owner's code; process isolation is not a security sandbox.
    with tempfile.TemporaryDirectory(prefix='ais-quiz-') as temp:
        env={k:v for k,v in os.environ.items() if k in ('PATH','SYSTEMROOT','LD_LIBRARY_PATH','PYTHONPATH')}
        env['PYTHONPATH']=os.pathsep.join(p for p in sys.path if p)
        env.update(HOME=temp,TMPDIR=temp,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        try:
            process=subprocess.run([sys.executable,str(ROOT/'quiz_app/worker.py')],
                input=json.dumps(dict(id=q['id'],answer=answer)),text=True,capture_output=True,cwd=temp,env=env,timeout=timeout)
            if process.returncode:
                return dict(status='unavailable',message='채점 프로세스 오류: '+process.stderr[-500:])
            return json.loads(process.stdout)
        except subprocess.TimeoutExpired:
            return dict(status='unavailable',message='실행 제한 시간(20초)을 초과했습니다. 무한 루프·너무 큰 연산을 확인하고 다시 제출하세요.')
        except (ValueError,OSError) as exc:
            return dict(status='unavailable',message='채점 결과를 읽지 못했습니다: '+str(exc)[:200])
