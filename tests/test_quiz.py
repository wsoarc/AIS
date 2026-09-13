import hashlib
from pathlib import Path
import json
import threading
import time
import unittest
from collections import Counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from quiz_app.catalog import BANK, BY_ID, FIELDS, assemble
from quiz_app.grading import check_in_process, grade
from quiz_app import server

class CatalogTests(unittest.TestCase):
    def test_in_scope_original_questions_preserved(self):
        hashes=json.loads(Path(__file__).with_name("original_question_hashes.json").read_text())
        self.assertEqual(len(hashes),26)
        excluded={'core-llm-classification-loss','core-llm-collate','core-llm-dpo',
                  'core-vis-evaluation','core-vis-training','core-vis-validation',
                  'core-rag-verification','core-od-activation','core-data-scaling',
                  'core-data-multistep','core-data-training'}
        self.assertTrue(excluded.isdisjoint(BY_ID))
        for id,expected in hashes.items():
            if id in excluded: continue
            actual=hashlib.sha256(json.dumps(BY_ID[id],sort_keys=True,ensure_ascii=True).encode()).hexdigest()
            self.assertEqual(actual,expected,id)

    def test_coverage_and_sources(self):
        self.assertEqual(len(BANK),86)
        for field in FIELDS:
            self.assertEqual({q['level'] for q in BANK if q['field']==field},{'easy','medium','hard'})
        for q in BANK:
            self.assertTrue(q['id'].startswith('core-'))
            self.assertEqual(q['template'].count('### 작성 필요 ###'),1)
            self.assertEqual(len(set(q['options'])),4)
            compile(assemble(q,q['answer']),'reference','exec')
            self.assertTrue(q['source']['path'].endswith('.ipynb'))
            self.assertEqual(len(q['source']['sha256']),64)
        self.assertIn('core-od-pruning',BY_ID)
        self.assertIn('core-od-speculative',BY_ID)
        self.assertEqual(len(BY_ID['core-od-pruning']['source']['related']),1)
        self.assertFalse({'od-range','vis-box','vis-rescale','rag-topk','rag-messages'} & BY_ID.keys())

    def test_reference_and_every_distractor(self):
        for q in BANK:
            with self.subTest(question=q['id'],kind='reference'):
                result=check_in_process(q,q['answer'])
                self.assertEqual(result.get('fraction'),1,result)
            for answer in q['options'][1:]:
                with self.subTest(question=q['id'],kind='distractor'):
                    result=check_in_process(q,answer)
                    self.assertEqual(result['status'],'graded',result)
                    self.assertLess(result['fraction'],1,result)

    def test_compact_mcq_blanks_and_distractors(self):
        for q in BANK:
            variant=q['mcq']
            with self.subTest(question=q['id']):
                self.assertEqual(assemble(variant,variant['answer']).strip(),assemble(q,q['answer']).strip())
                self.assertNotEqual(variant['template'],q['template'])
                self.assertEqual(len(set(variant['options'])),4)
                for i,option in enumerate(variant['options']):
                    self.assertLessEqual(len(option.splitlines()),3)
                    self.assertNotIn('def ',option)
                    self.assertNotIn('class ',option)
                    result=check_in_process(variant,option)
                    self.assertEqual(result['status'],'graded',result)
                    if i==0: self.assertEqual(result['fraction'],1,result)
                    else: self.assertLess(result['fraction'],1,result)

    def test_equivalent_implementations_and_partial_credit(self):
        alternatives={
            'core-llm-layernorm':'return F.layer_norm(x, (x.shape[-1],), self.scale, self.shift, self.eps)',
            'core-rag-pipeline':'chunks = self.retriever.retrieve(query, search_results, topk)\nreturn self.reader.generate_response(query, chunks), chunks',
            'core-od-speculative':'n = 0\nwhile n < gamma and drafted[0,n] == preds[n]:\n    n += 1\nreturn torch.cat((drafted[:,:n], preds[n:n+1].reshape(1,1)), dim=-1), n',
        }
        for id,answer in alternatives.items():
            self.assertEqual(check_in_process(BY_ID[id],answer)['fraction'],1,id)
        q=BY_ID['core-od-speculative']
        result=check_in_process(q,q['options'][1])
        self.assertGreater(result['fraction'],0)
        self.assertLess(result['fraction'],1)
        self.assertEqual(check_in_process(q,'this is not valid Python!')['fraction'],0)

    def test_subprocess_and_timeout(self):
        q=BY_ID['core-llm-layernorm']
        self.assertEqual(grade(q,q['answer'])['fraction'],1)
        result=grade(q,'while True:\n    pass',timeout=.1)
        self.assertEqual(result['status'],'unavailable')
        self.assertNotIn('score',result)

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True)
        cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.http.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close()
    def request(self,path,data=None,headers=None):
        headers=headers if headers is not None else {'X-Quiz-Token':server.TOKEN,'Content-Type':'application/json'}
        req=Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
        with urlopen(req) as r: return json.load(r)
    def test_exam_balance_secrecy_and_finish(self):
        s=self.request('/api/start',{'mode':'exam','kind':'mcq'})
        self.assertEqual(Counter(q['field'] for q in s['questions']),{f:4 for f in FIELDS})
        self.assertAlmostEqual(s['deadline']-s['started'],10800)
        for q in s['questions']:
            for key in ('answer','explanation','hint','order'):
                self.assertNotIn(key,q)
            qid=q['id']
            answer=str(server.SESSIONS[s['id']]['order'][qid].index(0))
            self.request('/api/save',{'session':s['id'],'id':qid,'answer':answer})
        qid=s['questions'][0]['id']
        with self.assertRaises(HTTPError) as err:
            self.request('/api/grade',{'session':s['id'],'id':qid,'answer':'0'})
        self.assertEqual(err.exception.code,400)
        self.assertFalse(self.request('/api/session?id='+s['id'])['results'])
        out=self.request('/api/finish',{'session':s['id']})
        self.assertTrue(out['finished']);self.assertFalse(out['busy'])
        self.assertTrue(all(r['fraction']==1 for r in out['results'].values()))
        self.assertEqual(self.request('/api/finish',{'session':s['id']})['results'],out['results'])
    def test_mcq_payload_and_revealed_answer_are_only_the_blank(self):
        s=self.request('/api/start',{'kind':'mcq','ids':['core-llm-attention']})
        public=s['questions'][0];q=BY_ID[public['id']]
        self.assertEqual(public['template'],q['mcq']['template'])
        self.assertNotEqual(public['template'],q['template'])
        self.assertTrue(all(len(o['code'].splitlines())<=3 for o in public['options']))
        choice=str(server.SESSIONS[s['id']]['order'][q['id']].index(0))
        result=self.request('/api/grade',{'session':s['id'],'id':q['id'],'answer':choice})
        self.assertEqual(result['answer'],q['mcq']['answer'])
        self.assertEqual(result['fraction'],1)
        code=self.request('/api/start',{'kind':'code','ids':[q['id']]})
        self.assertEqual(code['questions'][0]['template'],q['template'])
        self.assertFalse(code['questions'][0]['options'])

    def test_expiry_validation_and_origin(self):
        with self.assertRaises(HTTPError) as err:
            self.request('/api/start',{},headers={})
        self.assertEqual(err.exception.code,403)
        with self.assertRaises(HTTPError):
            self.request('/api/start',{},headers={'X-Quiz-Token':server.TOKEN,'Origin':'https://foreign.example'})
        s=self.request('/api/start',{'mode':'exam','kind':'mcq'})
        server.SESSIONS[s['id']]['deadline']=time.time()-1
        with self.assertRaises(HTTPError):
            self.request('/api/save',{'session':s['id'],'id':s['questions'][0]['id'],'answer':'0'})
        out=self.request('/api/finish',{'session':s['id']})
        self.assertTrue(out['finished'])
        self.assertTrue(all(r['score']==0 for r in out['results'].values()))
    def test_resume_and_changed_answer(self):
        s=self.request('/api/start',{'fields':['LLM'],'level':'easy','kind':'mcq','count':1})
        qid=s['questions'][0]['id'];body={'session':s['id'],'id':qid,'answer':'0'}
        r=self.request('/api/grade',body)
        self.assertEqual(r['status'],'graded')
        self.assertIn(qid,self.request('/api/session?id='+s['id'])['results'])
        self.request('/api/save',{**body,'answer':'1'})
        resumed=self.request('/api/session?id='+s['id'])
        self.assertNotIn(qid,resumed['results'])
        self.assertEqual(resumed['answers'][qid],'1')
        with self.assertRaises(HTTPError): self.request('/api/save',{**body,'answer':'9'})

if __name__=='__main__': unittest.main()
