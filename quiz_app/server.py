"""Loopback-only local study app. Run: python3 -m quiz_app"""
import argparse
import concurrent.futures
import importlib.util
import json
import random
import secrets
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from quiz_app.catalog import BANK, BY_ID, FIELDS, ROOT
from quiz_app.grading import grade

SESSIONS={}
LOCK=threading.RLock()
TOKEN=secrets.token_urlsafe(32)


def choose_questions(fields,level,count,rng):
    groups={f:[q for q in BANK if q['field']==f and (level=='mixed' or q['level']==level)] for f in fields}
    for group in groups.values(): rng.shuffle(group)
    chosen=[]
    while len(chosen)<count and any(groups.values()):
        for f in fields:
            if groups[f] and len(chosen)<count: chosen.append(groups[f].pop())
    rng.shuffle(chosen)
    return chosen


def public_question(q, options, kind):
    if kind == 'mcq': q = q['mcq']
    keys=['id','field','level','title','point','prompt','template','points','mode','cases']
    out={k:q[k] for k in keys}
    out.update(kind=kind,source={k:q['source'][k] for k in ('path','cell','symbol','related')},
               provided=q['setup'],fixture=q['fixture'],
               options=[{'id':str(i),'code':q['options'][original]} for i,original in enumerate(options)] if kind=='mcq' else [])
    return out


def create_session(data):
    mode=data.get('mode','practice')
    if mode not in ('practice','exam'): raise ValueError('잘못된 모드입니다.')
    kind=data.get('kind','mcq')
    level=data.get('level','mixed')
    if kind not in ('mcq','code','mixed') or level not in ('easy','medium','hard','mixed'): raise ValueError('유형·난이도를 확인하세요.')
    fields=data.get('fields',FIELDS)
    if not isinstance(fields,list) or not fields or any(f not in FIELDS for f in fields): raise ValueError('분야를 선택하세요.')
    fields=list(dict.fromkeys(fields))
    count=int(data.get('count',20))
    if count<1 or count>len(BANK): raise ValueError('문제 수 범위를 확인하세요.')
    rng=random.Random(secrets.randbits(64))
    if mode=='exam': fields,level,count=FIELDS,'mixed',20
    if data.get('ids') and mode=='practice':
        ids=data['ids']
        if not isinstance(ids,list) or len(ids)>len(BANK) or any(i not in BY_ID for i in ids): raise ValueError('복습 문제 목록을 확인하세요.')
        selected=[BY_ID[i] for i in dict.fromkeys(ids)];rng.shuffle(selected)
    else: selected=choose_questions(fields,level,count,rng)
    if not selected: raise ValueError('선택 조건에 해당하는 문제가 없습니다.')
    sid=secrets.token_urlsafe(24)
    questions=[]; order={}
    for i,q in enumerate(selected):
        options=list(range(4));rng.shuffle(options);order[q['id']]=options
        # Mixed sessions guarantee both answer types when there are >=2 items.
        qkind=('code' if i%2 else 'mcq') if kind=='mixed' else kind
        questions.append(public_question(q,options,qkind))
    now=time.time()
    s=dict(id=sid,mode=mode,questions=questions,order=order,answers={},results={},finished=False,busy=False,
           started=now,deadline=now+10800 if mode=='exam' else None,requested=count)
    SESSIONS[sid]=s
    return s


def view(s):
    return {k:s[k] for k in ('id','mode','questions','answers','results','finished','busy','started','deadline','requested')}


def evaluate(s,qid):
    q=BY_ID[qid]; answer=s['answers'].get(qid)
    kind=next(item['kind'] for item in s['questions'] if item['id']==qid)
    if kind == 'mcq': q = q['mcq']
    if answer is None or not str(answer).strip():
        result=dict(status='graded',fraction=0.,score=0.,max_score=q['points'],checks=[],message='미응답')
    elif kind=='mcq':
        correct=s['order'][qid][int(answer)]==0
        result=dict(status='graded',fraction=float(correct),score=q['points'] if correct else 0,max_score=q['points'],checks=[],message='정답' if correct else '오답')
    else: result=grade(q,answer)
    if result['status']=='graded': result.update(kind=kind,answer=q['answer'],explanation=q['explanation'],source=q['source'])
    return result


def safe_json(value):
    # JSON standard has no Infinity/NaN; grading values use readable strings.
    if isinstance(value,float) and (value!=value or abs(value)==float('inf')): return str(value)
    if isinstance(value,dict): return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,list): return [safe_json(v) for v in value]
    return value

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def respond(self,status,value,ctype='application/json; charset=utf-8'):
        body=(json.dumps(safe_json(value),ensure_ascii=False) if ctype.startswith('application/json') else value)
        if isinstance(body,str): body=body.encode()
        self.send_response(status)
        self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers(); self.wfile.write(body)
    def valid_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
    def do_GET(self):
        if not self.valid_host(): return self.respond(403,{'error':'로컬 주소로 접속하세요.'})
        url=urlparse(self.path)
        if url.path=='/api/meta':
            missing=[m for m in ('torch','numpy','sklearn','torchvision','PIL','pandas') if importlib.util.find_spec(m) is None]
            return self.respond(200,dict(token=TOKEN,fields=FIELDS,count=len(BANK),counts=dict(Counter(q['field'] for q in BANK)),
                coverage=[dict(field=q['field'],point=q['point'],level=q['level'],id=q['id'],title=q['title']) for q in BANK],missing=missing))
        if url.path=='/api/session':
            sid=parse_qs(url.query).get('id',[''])[0]
            with LOCK:
                s=SESSIONS.get(sid)
                return self.respond(200,view(s)) if s else self.respond(404,{'error':'이전 서버의 세션입니다. 새 퀴즈를 시작하세요. 복습 기록은 유지됩니다.'})
        files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if url.path in files:
            name=files[url.path];ctype={'html':'text/html','js':'text/javascript','css':'text/css'}[name.split('.')[-1]]+'; charset=utf-8'
            return self.respond(200,(ROOT/'quiz_app/static'/name).read_bytes(),ctype)
        return self.respond(404,{'error':'찾을 수 없습니다.'})
    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-Quiz-Token')!=TOKEN:
            return self.respond(403,{'error':'페이지를 새로고침한 뒤 다시 시도하세요.'})
        origin=self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
            return self.respond(403,{'error':'다른 출처의 요청은 허용하지 않습니다.'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=100000: raise ValueError('요청 크기를 확인하세요.')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict): raise ValueError('잘못된 요청입니다.')
            if self.path=='/api/start':
                with LOCK: s=create_session(data)
                return self.respond(200,view(s))
            with LOCK:
                s=SESSIONS.get(data.get('session'))
                if not s: return self.respond(404,{'error':'세션을 찾을 수 없습니다.'})
                if s['busy']: return self.respond(409,{'error':'채점 중입니다. 잠시 기다려 주세요.'})
                qid=data.get('id')
                if self.path in ('/api/save','/api/grade','/api/hint'):
                    if qid not in s['order']: raise ValueError('세션에 없는 문제입니다.')
                if self.path=='/api/hint':
                    if s['mode']=='exam' and not s['finished']: raise ValueError('모의시험에서는 종료 후 해설을 볼 수 있습니다.')
                    return self.respond(200,{'hint':BY_ID[qid]['hint']})
                if self.path=='/api/finish':
                    if s['finished']: return self.respond(200,view(s))
                    s['busy']=True
                elif self.path in ('/api/save','/api/grade'):
                    if s['finished']: raise ValueError('이미 종료된 퀴즈입니다.')
                    if s['deadline'] and time.time()>s['deadline']: raise ValueError('시험 시간이 종료되었습니다. 결과 보기를 눌러 주세요.')
                    if self.path=='/api/grade' and s['mode']=='exam': raise ValueError('모의시험은 종료 후 채점합니다.')
                    answer=data.get('answer','')
                    if not isinstance(answer,str) or len(answer)>16000: raise ValueError('답안은 16000자 이하 문자열이어야 합니다.')
                    kind=next(q['kind'] for q in s['questions'] if q['id']==qid)
                    if kind=='mcq' and answer not in ('','0','1','2','3'): raise ValueError('보기를 선택하세요.')
                    s['answers'][qid]=answer
                    # Editing a graded answer invalidates the previous score.
                    s['results'].pop(qid,None)
                    if self.path=='/api/save': return self.respond(200,{'saved':True})
                    s['busy']=True
                else: return self.respond(404,{'error':'찾을 수 없습니다.'})
            try:
                if self.path=='/api/grade':
                    result=evaluate(s,qid)
                    with LOCK: s['results'][qid]=result
                    return self.respond(200,result)
                pending=[q['id'] for q in s['questions'] if s['results'].get(q['id'],{}).get('status')!='graded']
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    results=list(pool.map(lambda qid:(qid,evaluate(s,qid)),pending))
                with LOCK:
                    s['results'].update(results)
                    s['finished']=all(r.get('status')=='graded' for r in s['results'].values())
                    s['busy']=False
                return self.respond(200,view(s))
            finally:
                with LOCK: s['busy']=False
        except (ValueError,TypeError,KeyError) as exc: self.respond(400,{'error':str(exc)})
        except Exception as exc: self.respond(500,{'error':'처리 오류: '+type(exc).__name__+': '+str(exc)[:200]})

def main():
    parser=argparse.ArgumentParser(description='AIS 실기 대비 로컬 퀴즈')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'AIS 퀴즈: http://127.0.0.1:{server.server_port} · {len(BANK)}문항 (종료: Ctrl+C)',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
