"""Curated notebook excerpts. Cell indices are zero based internally."""
import ast
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['LLM', 'Vision', 'On-device', 'Data', 'RAG']
MARKER = '### 작성 필요 ###'

class Clean(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node
    visit_AsyncFunctionDef = visit_FunctionDef

def normalized(s):
    return ast.unparse(Clean().visit(ast.parse(textwrap.dedent(s).strip())))

def cell(path, index):
    return ''.join(json.loads((ROOT / path).read_text())['cells'][index]['source'])

def extract(path, index, symbol=None, span=None):
    raw = cell(path, index)
    if symbol:
        tree = ast.parse(raw)
        for name in symbol.split('.'):
            tree = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name)
        return normalized(ast.get_source_segment(raw, tree))
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(span[0]))
    end = next(i for i in range(start, len(lines)) if lines[i].strip().startswith(span[1]))
    stop = end if len(span) > 2 and span[2] else end + 1
    return normalized('\n'.join(lines[start:stop]))

BANK = []
def add(id, field, level, title, point, path, index, *, prompt, fixture, symbol=None, span=None, signature=None, returns=None, hide=None, wrong, hint, explanation, setup='', mode='CPU 실행', cases=None, prefix=None, corrections=()):
    fragment = extract(path, index, symbol, span)
    for old,new in corrections:
        if fragment.count(old) != 1: raise ValueError((id,'source correction mismatch',old))
        fragment = fragment.replace(old,new)
    related = []
    if prefix:
        fragment = extract(prefix['path'], prefix['index'], span=prefix['span']) + '\n' + fragment
        related.append(dict(path=prefix['path'], cell=prefix['index']+1, symbol='연속 코드 발췌', sha256=hashlib.sha256(cell(prefix['path'], prefix['index']).encode()).hexdigest()))
    if signature is not None:
        declaration = 'async def' if any(isinstance(n, ast.Await) for n in ast.walk(ast.parse(fragment))) else 'def'
        reference = declaration + ' solve(' + signature + '):\n' + textwrap.indent(fragment, '    ')
        if returns: reference += '\n    return ' + returns
    else: reference = fragment
    reference = normalized(reference)
    if hide is None:
        fn = ast.parse(reference).body[0]
        answer = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))
    else:
        answer = normalized(hide)
    # Locate a whole, consistently indented block in normalized source.
    matches = []
    for width in range(0, 21, 4):
        target = textwrap.indent(answer, ' ' * width)
        if target in reference: matches.append((width, target))
    if not matches: raise ValueError((id, 'mask missing', answer, reference))
    width, target = max(matches)
    if reference.count(target) != 1: raise ValueError((id, 'ambiguous mask'))
    template = reference.replace(target, ' ' * width + MARKER)
    distractors = []
    for old, new in wrong:
        if old not in answer: raise ValueError((id, 'wrong replacement missing', old, answer))
        distractors.append(answer.replace(old, new))
    if len(set([answer] + distractors)) != 4: raise ValueError((id, 'duplicate options'))
    BANK.append(dict(id=id, field=field, level=level, title=title, point=point,
        prompt=prompt, template=template, answer=answer, indent=width,
        options=[answer]+distractors, hint=hint, explanation=explanation,
        setup=setup, fixture=textwrap.dedent(fixture), mode=mode,
        cases=cases or ['기본 입력', '다른 크기·값', '추가 조건'],
        source=dict(path=path, cell=index+1, symbol=symbol or '연속 코드 발췌',
                    sha256=hashlib.sha256(cell(path,index).encode()).hexdigest(), related=related,
                    **({'corrections':list(corrections)} if corrections else {})),
        points={'easy':5,'medium':10,'hard':15}[level]))

LLM='1. LLM/example_ipython/'
VIS='2. Vision/example_ipython/'
OD='3. On-device/example_ipython/'
DATA='4. Data/'
RAG='5. RAG/실습자료_Colab/'

add('core-llm-attention', 'LLM', 'hard', 'Causal Attention: 토큰 간 정보 흐름', '모델의 원리 이해', LLM + 'Chapter_3_Excercise_Attention.ipynb', 0,
    symbol='CausalAttention.forward',
    prompt='x는 (B,T,D_in)입니다. 제공된 Q/K/V 투영, self.mask, dropout으로 인과적 attention을 구현하세요. 출력은 (B,T,D_out)이고 미래 토큰을 참조하면 안 됩니다. 검증에서는 dropout=0입니다.',
    fixture='d=4+case\nx=torch.randn(2,3+case,d)\nself=NS(W_key=nn.Linear(d,4), W_query=nn.Linear(d,4), W_value=nn.Linear(d,4), mask=torch.triu(torch.ones(8,8),diagonal=1),dropout=nn.Identity())\nresult=forward(self,x)',
    wrong=[('-torch.inf', '0.0'), ('dim=-1', 'dim=1'), ('attn_weights @ values', 'attn_weights @ keys')],
    hint='QKᵀ → 미래 마스킹 → 스케일링·softmax → V 가중합 순서입니다.',
    explanation='마스크 위치를 -inf로 채워 미래 확률을 0으로 만들고, key 차원의 제곱근으로 점수를 스케일링합니다.',
)

add('core-llm-layernorm', 'LLM', 'easy', 'LayerNorm: 토큰별 정규화와 학습 파라미터', '모델의 원리 이해', LLM + 'Chapter_4_Excercise_GPT.ipynb', 2,
    symbol='LayerNorm.forward',
    fixture='x = torch.randn(2, 3+case, 4)\nself = NS(eps=1e-5, scale=torch.tensor([1.,2.,3.,4.]), shift=torch.tensor([.5,-.5,1.,2.]))\nresult = forward(self, x)',
    wrong=[('x - mean', 'x + mean'), ('torch.sqrt(var + self.eps)', '(var + self.eps)'), ('+ self.shift', '- self.shift')],
    hint='표준편차로 나누고, 학습 가능한 affine 변환을 적용합니다.',
    explanation='노트북은 마지막 차원의 모집단 분산을 사용합니다. scale과 shift도 출력에 반영해야 합니다.',
    prompt='입력 x는 (B,T,D)입니다. 각 토큰의 마지막 특성 축을 정규화하고 학습 가능한 scale과 shift를 적용하는 forward 전체를 구현하세요. self에는 eps, scale, shift가 제공됩니다. 배치나 토큰끼리 통계를 섞지 않아야 합니다.',
)

add('core-llm-training', 'LLM', 'medium', '언어 모델 학습: 손실에서 가중치 갱신까지', '모델의 원리 이해', LLM + 'Chapter_5_Excercise_Pretraining.ipynb', 2,
    span=('optimizer.zero_grad()', 'optimizer.step()'),
    signature='optimizer, input_batch, target_batch, model, device',
    returns='loss',
    prompt='제공된 calc_loss_batch를 사용하여 한 배치를 학습하세요. 이전 기울기를 초기화하고 역전파 후 optimizer로 갱신하세요.',
    setup=extract(LLM + 'Chapter_5_Excercise_Pretraining.ipynb', 1, 'calc_loss_batch'),
    fixture='model=nn.Embedding(7,7)\noptimizer=torch.optim.AdamW(model.parameters(),lr=.01)\nx=torch.randint(0,7,(2,3+case)); y=torch.randint(0,7,x.shape)\nfor p in model.parameters(): p.grad=torch.ones_like(p)\nloss=solve(optimizer,x,y,model,"cpu")\nresult=(loss,[p.detach() for p in model.parameters()],[p.grad for p in model.parameters()])',
    wrong=[('optimizer.zero_grad()', 'pass'), ('loss.backward()', 'pass'), ('optimizer.step()', 'pass')],
    hint='이전 gradient 제거 → 손실 계산 → backward → step입니다.',
    explanation='학습 대상 모델의 예측으로 손실을 계산하고, 이전 기울기 초기화·역전파·파라미터 갱신을 연결합니다. calc_loss_batch도 원본 노트북에서 제공합니다.',
)

add('core-vis-evaluation', 'Vision', 'easy', '분류 평가: 혼동행렬과 클래스별 성능', '평가 Metric 이해', VIS + '01_ResNet18_CIFAR10.ipynb', 16,
    symbol='confusion_and_perclass',
    prompt='혼동행렬은 행=정답, 열=예측입니다. int64 혼동행렬과 클래스별 정확도를 반환하세요. 정답 샘플이 없는 클래스의 정확도는 0으로 처리합니다.',
    fixture='result=confusion_and_perclass(torch.tensor([0,1,1,2,case]),torch.tensor([0,0,1,1,0]),4)',
    wrong=[('cm[int(t), int(p)]', 'cm[int(p), int(t)]'), ('cm.sum(dim=1)', 'cm.sum(dim=0)'), ('min=1.0', 'min=0.0')],
    hint='대각 원소를 각 정답 클래스의 샘플 수로 나눕니다.',
    explanation='빈 클래스의 분모를 최소 1로 제한합니다. 정답 축과 예측 축이 바뀌지 않아야 합니다.',
)

add('core-data-window', 'Data', 'easy', '시계열 데이터: 과거 구간과 미래 정답 구성', '데이터 구성 원리 이해', DATA + '3_4-ts-practice/ts_solution.ipynb', 15,
    symbol='convert_data_into_tensors',
    prompt='data_seq는 (N,1) NumPy 배열, sequence_length는 제공된 양의 정수입니다. 각 윈도우와 그 직후의 값을 float32 텐서로 반환하세요. N > sequence_length입니다.',
    fixture='sequence_length=3+case\nresult=convert_data_into_tensors(np.arange(10+case,dtype=float).reshape(-1,1))',
    wrong=[('i + sequence_length, 0', 'i + sequence_length - 1, 0'), ('dtype=torch.float32', 'dtype=torch.float64'), ('len(data_seq) - sequence_length', 'len(data_seq) - sequence_length - 1')],
    hint='윈도우는 [i:i+L], 정답은 i+L 위치입니다.',
    explanation='입력 shape은 (N-L,L,1), 정답 shape은 (N-L,)이며 float32로 반환합니다.',
)

add('core-data-scaling', 'Data', 'easy', '학습·평가 데이터의 일관된 스케일링', '관련 Library 활용', DATA + '3_4-ts-practice/ts_solution.ipynb', 13,
    span=('scaler =', 'test_scaled ='),
    signature='train_data, test_data',
    returns='train_scaled, test_scaled',
    setup='from sklearn.preprocessing import MinMaxScaler',
    fixture='result=solve(NS(values=np.array([[0.],[5.],[10.]])),NS(values=np.array([[2.+case],[12.+case]])))',
    wrong=[('scaler.transform', 'scaler.fit_transform'), ('test_data.values', 'train_data.values'), ('scaler.transform(test_data.values)', 'test_data.values')],
    hint='평가 데이터에는 transform만 적용합니다.',
    explanation='test 값이 학습 범위를 벗어나면 변환값이 1보다 클 수 있습니다. 별도 fit은 데이터 누수 및 서로 다른 척도를 만듭니다.',
    prompt='학습과 평가에 같은 척도를 적용하도록 MinMaxScaler로 데이터 준비 과정을 구현하세요. train_data/test_data의 .values는 (N,F) 배열입니다. scaler 학습에 평가 데이터가 섞이지 않아야 하며 변환된 두 배열을 반환하세요.',
)

add('core-data-bpr', 'Data', 'hard', '추천 모델: 선호 순위를 학습하는 BPR 손실', '모델의 원리 이해', DATA + '7_8-recsys-practice/RecSys_GCF_sol.ipynb', 14,
    symbol='NGCF.bpr_loss',
    prompt='user_emb, pos_item_emb, neg_item_emb는 (B,D)입니다. positive 점수가 negative보다 높아지도록 BPR 손실과 원본의 L2 정규화를 계산하세요.',
    fixture='u=torch.randn(3+case,4,requires_grad=True); p=torch.randn_like(u); n=torch.randn_like(u)\nloss=bpr_loss(None,u,p,n,reg_weight=.01);loss.backward()\nresult=(loss,u.grad)',
    wrong=[('pos_scores - neg_scores', 'neg_scores - pos_scores'), ('return loss + reg_loss', 'return loss'), ('dim=1', 'dim=0')],
    hint='-log sigmoid(positive−negative)의 평균을 사용합니다.',
    explanation='세 임베딩의 제곱 L2 norm을 배치 크기로 나눈 정규화 항을 더합니다.',
)

add('core-od-perchannel', 'On-device', 'medium', '레이어의 출력 채널별 가중치 양자화', 'Quantization', OD + '1. Basic Quantization Lab_answer.ipynb', 43,
    symbol='linear_quantize_weight_per_channel',
    setup='\n\n'.join((extract(OD + '1. Basic Quantization Lab_answer.ipynb', i, name) for (i, name) in [(22, 'get_quantized_range'), (24, 'linear_quantize'), (39, 'get_quantization_scale_for_weight')])),
    fixture='shape=[(3,4),(3,2,2,2),(2,5)][case]\nx=torch.arange(1,1+math.prod(shape),dtype=torch.float32).reshape(shape)\nresult=linear_quantize_weight_per_channel(x,4)',
    wrong=[('scale_shape[dim_output_channels] = -1', 'scale_shape[-1] = -1'), ('scale = scale.view(scale_shape)', 'scale = scale'), ('[1] * tensor.dim()', '[1] * (tensor.dim() + 1)')],
    hint='출력 채널 축만 O, 나머지 축은 1이어야 합니다.',
    explanation='Linear에는 (O,1), Conv에는 (O,1,1,1)의 scale을 적용합니다. 양자화 결과와 scale shape을 함께 검증합니다.',
    prompt='Linear (O,I) 또는 Conv (O,I,H,W)의 실수 가중치를 출력 채널마다 다른 scale로 양자화하는 함수 전체를 구현하세요. 반환값은 (양자화 가중치, 브로드캐스트 가능한 scale, zero_point=0)입니다. 범위·scale·양자화 연산 헬퍼는 제공됩니다.',
)

add('core-od-pruning', 'On-device', 'hard', 'KV Cache Pruning: 중요도 선택과 메모리 축소', 'Pruning · KV Cache', OD + '4_[Colab]_KV_Cache_Pruning_answer.ipynb', 15,
    span=('keep = scores.topk', 'layer.values ='),
    signature='attn_kv, layer, budget',
    returns='layer.keys, layer.values',
    prompt='H2O 방식으로 토큰 중요도를 계산한 뒤 KV cache를 축소하세요. attn_kv는 (B,H,Q,K), keys/values는 (B,H,K,D)입니다. 모든 query의 attention을 누적하고 최근 budget//2개 토큰은 항상 보존합니다. 상위 budget개를 원래 위치 순서로 정렬해 K와 V에 함께 적용하세요. budget은 2 이상입니다.',
    fixture='attn_kv=torch.tensor([[.7,.05,.1,.06,.05,.04],[.6,.1,.1,.1,.06,.04],[.03,.08,.61,.12,.11,.05]]).repeat(2,2,1,1)\nlayer=NS(keys=torch.arange(72.).reshape(2,2,6,3),values=torch.arange(72.).reshape(2,2,6,3)*2)\nresult=solve(attn_kv,layer,2+case)',
    wrong=[('.indices.sort(-1).values', '.indices'), ('layer.values.gather(2, idx)', 'layer.keys.gather(2, idx)'), ('.gather(2, idx)', '.gather(1, idx)')],
    prefix={'path':OD+'4_[Colab]_KV_Cache_Pruning_answer.ipynb','index':20,'span':('recent_size =','scores[..., -recent_size:]')},
    hint='topk → 위치 정렬 → index 차원 확장 → gather 순서입니다.',
    explanation='query별 attention 누적으로 중요도를 판단하고 최근 토큰을 보호한 뒤, 동일한 토큰 선택을 K와 V에 적용합니다. 같은 노트북의 H2O 점수 함수와 cache 축소 구간을 연결한 문제입니다.',
)

add('core-od-speculative', 'On-device', 'hard', 'Speculative Decoding: Target 검증과 수락 알고리즘', 'Speculative Decoding · 추가 범위', OD + '5_[Colab]Speculative_Decoding_Answer.ipynb', 8,
    symbol='accept_draft',
    prompt='drafted는 (1,gamma), preds는 길이 gamma+1입니다. 앞에서부터 일치하는 후보만 수락하고, 첫 불일치 위치의 Target 토큰을 붙이세요. 모두 일치하면 마지막 bonus token을 붙입니다. (new_tokens,n)을 반환하세요.',
    fixture='drafted=torch.tensor([[2,3,4]])\npreds=torch.tensor([[8,3,4,5],[2,8,4,5],[2,3,4,5]][case])\nresult=accept_draft(drafted,preds,3)',
    cases=['첫 후보부터 불일치', '중간에서 불일치', '모두 일치·bonus token'],
    wrong=[('break', 'continue'), ('drafted[:, :n]', 'drafted[:, :gamma]'), ('preds[n]', 'preds[0]')],
    hint='일치 개수 전체가 아니라 일치하는 연속 prefix 길이를 셉니다.',
    explanation='첫 mismatch 이후 다시 일치해도 수락하지 않습니다. 전부 수락하면 preds[gamma]를 추가합니다.',
)

add('core-rag-mcp', 'RAG', 'medium', 'MCP RAG: 비동기 검색과 생성의 전체 흐름', '관련 Library 활용', RAG + '2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb', 78,
    symbol='RAGwithMCP.inference',
    prompt='self.retrieve(query,query_time,search_results,topk)는 async 함수입니다. 검색 결과를 동기 generate_response(query,query_time,retrieved_results)에 전달하고, retrieved_results와 answer 키를 가진 dict를 반환하세요.',
    fixture='calls=[]\nasync def retrieve(*args):\n    calls.append(("retrieve",args));return ["evidence-"+str(case)]\ndef generate_response(*args):\n    calls.append(("generate",args));return "answer"\nself=NS(retrieve=retrieve,generate_response=generate_response)\nresult=(asyncio.run(inference(self,"q",["search"],"2026-08-01",case+1)),calls)',
    wrong=[('await self.retrieve', 'self.retrieve'), ('self.retrieve(query, query_time, search_results, topk)', 'self.retrieve(query, search_results, query_time, topk)'), ("'answer': answer", "'answer': retrieved_results")],
    hint='검색은 await, 응답 생성은 동기 호출입니다.',
    explanation='query_time과 search_results 순서를 지키며 검색 결과를 생성 단계로 연결해야 합니다.',
    mode='비동기 호출 실행 · MCP/Reader 대체 객체',
)


add('core-llm-dataset','LLM','medium','언어 모델 데이터셋: 슬라이딩 윈도우와 next-token 학습','데이터 구성 원리 이해',LLM+'Chapter_2_Exercise_Dataset.ipynb',2,
    symbol='GPTDatasetV1',
    hide='for i in range(0, len(token_ids) - max_length, stride):\n    input_chunk = token_ids[i:i + max_length]\n    target_chunk = token_ids[i + 1:i + max_length + 1]\n    self.input_ids.append(torch.tensor(input_chunk))\n    self.target_ids.append(torch.tensor(target_chunk))',
    setup='from torch.utils.data import Dataset',
    prompt='토큰화된 문서를 GPT 학습용 데이터셋으로 구성하세요. max_length 길이의 입력과 다음 토큰 정답을 stride 간격으로 생성하고, 같은 길이의 정수 텐서 쌍을 저장해야 합니다. 토크나이저와 Dataset 인터페이스는 제공됩니다.',
    fixture='tokenizer=NS(encode=lambda txt,**kwargs:list(range(11)))\ndataset=GPTDatasetV1("fixture",tokenizer,3+case,1+case)\nresult=(len(dataset),[dataset[i] for i in range(len(dataset))])',
    wrong=[('i + 1:i + max_length + 1','i:i + max_length'), ('max_length, stride','max_length, max_length'), ('self.target_ids.append(torch.tensor(target_chunk))','self.target_ids.append(torch.tensor(input_chunk))')],
    hint='각 입력 위치의 정답은 한 토큰 뒤이며, 윈도우 이동 간격은 길이와 별개의 값입니다.',
    explanation='next-token 학습에서는 입력·타깃의 위치 정렬이 핵심입니다. 윈도우 수, 각 샘플의 shape·dtype·정답 대응을 함께 검증합니다.')

add('core-llm-transformer','LLM','hard','Transformer block: 정규화·Attention·FFN·잔차 연결','모델의 원리 이해',LLM+'Chapter_4_Excercise_GPT.ipynb',3,
    symbol='TransformerBlock.forward',
    prompt='노트북의 pre-norm Transformer block 전체 forward를 구현하세요. self.norm1/norm2, att, ff, drop_shortcut이 제공됩니다. Attention과 FFN 두 단계에서 입력 정보를 잔차로 유지하고 최종 (B,T,D) 텐서를 반환해야 합니다.',
    fixture='self=NS(norm1=nn.LayerNorm(4),norm2=nn.LayerNorm(4),att=nn.Linear(4,4),ff=nn.Sequential(nn.Linear(4,8),nn.GELU(),nn.Linear(8,4)),drop_shortcut=nn.Identity())\nx=torch.randn(2,3+case,4,requires_grad=True)\nout=forward(self,x);out.sum().backward()\nresult=(out,x.grad)',
    wrong=[('x = x + shortcut','x = x'), ('self.norm2(x)','self.norm1(x) * 0'), ('self.ff(x)','self.att(x)')],
    hint='각 하위 블록에 들어가기 전의 입력을 별도로 보관해야 합니다.',
    explanation='단순한 레이어 호출이 아니라 두 잔차 경로와 pre-norm의 구성 원리를 평가합니다. 작은 제공 레이어로 출력 및 입력 기울기 흐름을 검증합니다.')

add('core-vis-preprocessing','Vision','easy','이미지 데이터: 학습 증강과 평가 전처리','데이터 구성 원리 이해 · 관련 Library 활용',VIS+'01_ResNet18_CIFAR10.ipynb',5,
    span=('train_tfms =','data_root =',True),signature='',returns='train_tfms, test_tfms',
    hide='train_tfms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])\ntest_tfms = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])',
    setup='from torchvision import transforms as T\nfrom PIL import Image\nCIFAR10_MEAN=(0.4914,0.4822,0.4465)\nCIFAR10_STD=(0.2023,0.1994,0.2010)',
    prompt='CIFAR-10의 학습·평가 전처리를 함께 구성하세요. 학습에는 노트북의 crop(32,padding=4)과 좌우 반전 증강을 적용하고, 평가는 증강 없이 재현 가능한 입력을 만듭니다. 두 경로 모두 RGB 이미지를 (C,H,W) float 텐서로 바꾸고 같은 평균·표준편차로 정규화해야 합니다. 주어진 증강 설정 수치 자체는 암기 대상이 아닙니다.',
    fixture='img=Image.fromarray((np.arange(32*32*3).reshape(32,32,3)+case*17).astype(np.uint8))\ntrain_tfms,test_tfms=solve()\nresult=(train_tfms(img),test_tfms(img),test_tfms(img))',
    wrong=[('train_tfms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),','train_tfms = T.Compose(['), ('test_tfms = T.Compose([T.ToTensor(),','test_tfms = T.Compose([T.RandomHorizontalFlip(p=1), T.ToTensor(),'), ('T.Normalize(CIFAR10_MEAN, CIFAR10_STD)','T.Normalize((0.,0.,0.), (1.,1.,1.))')],
    hint='학습의 다양성과 평가의 일관성을 구분하고 두 경로의 입력 척도를 맞춥니다.',
    explanation='transforms 개별 이름을 맞히는 대신 학습·평가 목적에 맞는 전체 파이프라인을 평가합니다. 동일한 seed로 실제 이미지 변환을 검증합니다.')

add('core-vis-vit','Vision','hard','ViT: 이미지 패치에서 분류 출력까지','모델의 원리 이해',VIS+'02_ViT_CIFAR10.ipynb',10,
    symbol='ViT.forward',
    setup='def repeat(t, pattern, b):\n    return t.expand(b, -1, -1)',
    prompt='이미지 입력부터 클래스 출력까지 ViT의 forward를 구현하세요. 패치 임베딩, cls_token, pos_embedding, dropout, transformer, mlp_head가 제공됩니다. pool="cls"이면 CLS 표현, 그 외에는 토큰 평균을 사용합니다. 최종 출력은 (B,클래스 수)입니다. repeat는 배치 방향 확장 헬퍼로 제공됩니다.',
    fixture='class PatchEmbedding(nn.Module):\n    def __init__(self):\n        super().__init__();self.proj=nn.Conv2d(3,4,2,2)\n    def forward(self,x):\n        return self.proj(x).flatten(2).transpose(1,2)\nself=NS(to_patch_embedding=PatchEmbedding(),cls_token=torch.randn(1,1,4),pos_embedding=torch.randn(1,17,4),dropout=nn.Identity(),transformer=nn.Linear(4,4),mlp_head=nn.Linear(4,3),pool="cls" if case!=1 else "mean")\nresult=forward(self,torch.randn(2,3,4+case*2,4+case*2))',
    wrong=[('x = x + self.pos_embedding[:, :n + 1]','x = x'), ('x[:, 0]','x[:, -1]'), ('torch.cat((cls_tokens, x), dim=1)','torch.cat((x, cls_tokens), dim=1)')],
    hint='패치 토큰에 CLS와 위치 정보를 넣은 뒤 Transformer의 출력 표현을 분류기로 전달합니다.',
    explanation='CLS의 위치, 위치 임베딩, 토큰 축과 pooling을 포함한 전체 텐서 흐름을 검증합니다. 제공 레이어는 흐름 검증용 작은 모듈이며 전체 ViT 학습 성능을 평가하지 않습니다.')

add('core-vis-unet','Vision','hard','U-Net: Encoder·Skip·Decoder의 전체 연결','모델의 원리 이해',VIS+'04_Unet.ipynb',6,
    symbol='UNet.forward',
    setup='\n\n'.join(extract(VIS+'04_Unet.ipynb',6,s) for s in ['DoubleConv','Down','Up','OutConv']),
    prompt='제공된 U-Net 블록으로 전체 forward를 구현하세요. inc와 down1~4로 다중 해상도 특징을 만들고, up1~4에 대응하는 encoder 특징을 skip으로 전달해 해상도를 복원합니다. outc의 출력은 입력과 같은 공간 크기의 분할 logits입니다.',
    fixture='self=NS(inc=DoubleConv(1,2),down1=Down(2,4),down2=Down(4,8),down3=Down(8,16),down4=Down(16,32),up1=Up(32,16),up2=Up(16,8),up3=Up(8,4),up4=Up(4,2),outc=OutConv(2,1))\nx=torch.randn(1,1,32+case*16,32+case*16)\nresult=forward(self,x)',
    wrong=[('self.up1(x5, x4)','self.up1(x5, x3)'), ('self.up4(x, x1)','self.up4(x, x2)'), ('return logits','return x')],
    hint='디코더 단계마다 해상도가 대응하는 encoder 특징을 연결합니다.',
    explanation='concat 한 줄이 아닌 encoder–decoder 구조 전체를 평가합니다. 원본 블록의 채널 수만 줄인 CPU 입력으로 실행합니다.')

add('core-vis-training','Vision','medium','분할 모델 학습: 예측과 마스크를 학습으로 연결','모델의 원리 이해',VIS+'04_Unet.ipynb',10,
    symbol='train_one_epoch',hide='optimizer.zero_grad()\nlogits = model(imgs)\nloss = criterion(logits, masks)\nloss.backward()\noptimizer.step()',
    prompt='분할 모델의 학습 루프에서 핵심 한 배치 학습 구간을 완성하세요. imgs와 정답 masks는 (B,1,H,W), 모델 출력은 sigmoid 이전 logits입니다. 제공된 BCEWithLogitsLoss를 이용해 분할 모델의 파라미터를 학습시키세요.',
    fixture='model=nn.Conv2d(1,1,3,padding=1)\noptimizer=torch.optim.Adam(model.parameters(),lr=.01)\nx=torch.randn(5,1,4+case,4+case);y=torch.randint(0,2,x.shape).float()\nloader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=2)\nresult=(train_one_epoch(model,loader,optimizer,nn.BCEWithLogitsLoss(),"cpu"),[p.detach() for p in model.parameters()])',
    wrong=[('optimizer.zero_grad()','pass'), ('loss.backward()','pass'), ('criterion(logits, masks)','criterion(masks, logits)')],
    hint='예측·정답의 대응을 유지하고 배치마다 기울기를 초기화한 뒤 갱신합니다.',
    explanation='이미지 입력과 정답 마스크가 loss 및 파라미터 갱신에 연결되는지 확인합니다. 학습된 가중치와 샘플 가중 평균 손실을 검증합니다.')

add('core-data-lstm','Data','medium','시계열 예측 모델: LSTM 구성과 텐서 흐름','모델의 원리 이해',DATA+'3_4-ts-practice/ts_solution.ipynb',21,
    symbol='LSTMModel',
    prompt='입력 (B,T,input_size)을 받아 모든 시점의 예측 (B,T,1)을 반환하는 LSTM 모델 클래스를 구현하세요. hidden_size와 num_layers가 주어집니다. 마지막 시점 선택은 학습·평가 쪽에서 수행하므로 모델은 시간 축을 유지해야 합니다.',
    fixture='model=LSTMModel(2,4+case,2)\nx=torch.randn(2,5+case,2,requires_grad=True)\nout=model(x);out.sum().backward()\nresult=(out,x.grad)',
    wrong=[('batch_first=True','batch_first=False'), ('return self.linear(out)','return self.linear(out[:, -1])'), ('nn.Linear(hidden_size, 1)','nn.Linear(hidden_size, hidden_size)')],
    hint='순환층이 반환하는 시퀀스 특징을 각 시점의 예측값으로 투영합니다.',
    explanation='생성자와 forward 전체를 구현해 모델 구조·배치/시간 축·예측층 연결을 평가합니다.')

add('core-data-training','Data','hard','시계열 학습: 예측 시점·정답·손실·역전파','모델의 원리 이해',DATA+'3_4-ts-practice/ts_solution.ipynb',23,
    span=('pred = model','optimizer.step()'),signature='model, batch_x, batch_y, loss_fn, optimizer',returns='loss',
    prompt='모델은 (B,T,1)을 출력하고 batch_y는 윈도우 직후의 정답 (B,)입니다. 학습 목적에 맞는 시점의 예측을 선택하고, 제공된 MSE 손실과 optimizer로 한 배치 학습을 수행하세요.',
    fixture='model=nn.Linear(1,1)\noptimizer=torch.optim.Adam(model.parameters(),lr=.01)\nx=torch.randn(3,4+case,1);y=torch.randn(3)\nfor p in model.parameters():p.grad=torch.ones_like(p)\nloss=solve(model,x,y,nn.MSELoss(),optimizer)\nresult=(loss,[p.detach() for p in model.parameters()],[p.grad for p in model.parameters()])',
    wrong=[('[:, -1, 0]','[:, 0, 0]'), ('optimizer.zero_grad()','pass'), ('loss.backward()','pass')],
    hint='윈도우의 마지막 출력이 다음 시점 정답과 대응합니다.',
    explanation='텐서 인덱스를 외우는 문제가 아니라 어떤 출력이 학습 목표와 대응하는지, 그 손실이 가중치 갱신으로 이어지는지 검증합니다.')

add('core-data-evaluation','Data','easy','시계열 평가: 추론 결과와 RMSE·MAPE','평가 Metric 이해 · 관련 Library 활용',DATA+'3_4-ts-practice/ts_solution.ipynb',27,
    symbol='test',
    setup='device="cpu"\nfrom sklearn.metrics import mean_squared_error, mean_absolute_percentage_error',
    prompt='모델의 (N,T,1) 출력과 윈도우 다음 시점의 정답 y_test로 평가 함수를 구현하세요. 평가 모드와 gradient 비활성화로 예측하고 RMSE·MAPE를 반환합니다. sklearn MAPE는 100을 곱하지 않은 비율입니다.',
    fixture='model=nn.Linear(1,1)\nx=torch.arange(1.,13.).reshape(3,4,1)+case;y=torch.tensor([3.,6.,9.])\nresult=(test(model,x,y),model.training)',
    wrong=[('y_hat[:, -1, 0]','y_hat[:, 0, 0]'), ('np.sqrt(mean_squared_error(y_test, test_predictions))','mean_squared_error(y_test, test_predictions)'), ('model.eval()','model.train()')],
    hint='학습에서 정답과 대응시킨 동일 시점의 출력을 평가해야 합니다.',
    explanation='평가 모드, 예측·정답 대응, 두 지표의 의미를 하나의 함수로 확인합니다.')

add('core-od-quantization','On-device','easy','가중치 양자화·역양자화의 전체 과정','Quantization',OD+'2. Advanced Quantization Lab_answer.ipynb',17,
    symbol='pseudo_quantize_tensor',
    prompt='실수 가중치 행렬을 n_bit 정수 범위로 양자화한 뒤 실수로 복원하는 함수 전체를 구현하세요. 행별 min/max로 scale과 zero point를 정하고 rounding·clipping·역양자화를 연결합니다. q_group_size>0이면 해당 크기로 그룹화하고 반환 시 원래 shape을 복원합니다. scale 안정화를 위한 최소 범위 1e-5는 제공된 조건입니다.',
    fixture='w=torch.tensor([[-2.,-1.,.2,1.],[-.3,.5,.7,1.8]])+case*.13\nresult=pseudo_quantize_tensor(w,n_bit=3+case,q_group_size=2 if case==1 else -1)',
    wrong=[('(w - zeros) * scales','(w + zeros) * scales'), ('torch.round(w / scales)','torch.round(w * scales)'), ('max_int = 2 ** n_bit - 1','max_int = 2 ** n_bit')],
    hint='실수→정수 코드→복원 실수의 두 방향 변환에서 같은 scale과 zero point를 사용합니다.',
    explanation='정수 범위 상수만 묻지 않고 레이어 가중치의 범위 추정부터 양자화 오차를 포함한 복원까지 평가합니다. 모델 다운로드 없는 simulated quantization입니다.')

add('core-od-distillation','On-device','hard','지식증류: Teacher 지식과 정답으로 Student 학습','Distillation',OD+'3. Knowledge Distillation_answer.ipynb',26,
    span=('with torch.no_grad():','optimizer.step()'),signature='teacher, student, inputs, labels, T, soft_target_loss_weight, ce_loss_weight, optimizer',returns='loss',
    setup='ce_loss=nn.CrossEntropyLoss()',
    prompt='Teacher의 예측 분포와 실제 정답을 함께 사용해 Student를 한 배치 학습시키세요. Teacher는 eval, Student는 train 상태이며 optimizer의 기울기는 이미 초기화되어 있습니다. 온도 T를 적용한 KL(Teacher||Student)에 T²를 반영하고 정답 cross entropy와 가중합합니다. Teacher에는 gradient를 남기지 않아야 합니다.',
    fixture='teacher=nn.Linear(4,3);student=nn.Linear(4,3);teacher.eval();student.train()\noptimizer=torch.optim.Adam(student.parameters(),lr=.01);optimizer.zero_grad()\nx=torch.randn(3+case,4);labels=torch.arange(3+case)%3\nloss=solve(teacher,student,x,labels,2.+case,.7,.3,optimizer)\nresult=(loss,[p.grad for p in teacher.parameters()],[p.detach() for p in student.parameters()],[p.grad for p in student.parameters()])',
    wrong=[('with torch.no_grad():','with torch.enable_grad():'), ('T ** 2','T'), ('optimizer.step()','pass')],
    hint='Teacher는 고정된 soft target을 제공하고, 두 손실의 gradient는 Student에만 전달합니다.',
    explanation='손실 공식만 계산하는 수준에서 확장해 Teacher 고정, Student 예측, soft/hard 손실 결합과 실제 학습 갱신까지 평가합니다.')

add('core-rag-retriever','RAG','hard','Retriever: 문서·청크·임베딩·유사도 검색','RAG 구성 이해 및 활용',RAG+'2일차/code/2. Task_1.ipynb',16,
    symbol='BaseRetriever.retrieve',
    setup='def parse_htmls(results):\n    return results\ndef extract_chunks(documents):\n    return [chunk for document in documents for chunk in document]',
    prompt='검색 결과 문서를 청크로 나누고, 문서·질문 임베딩의 cosine similarity로 관련 청크를 선택하는 Retriever 전체를 구현하세요. parse_htmls, extract_chunks, self.embed_text는 제공됩니다. 0 벡터와 유사도 동률은 없으며 반환 순서는 높은 유사도 순입니다.',
    fixture='vectors={"query":[1.,0.],"a":[1.,0.],"b":[3.,4.],"c":[-1.,0.],"d":[1.,2.]}\ncalls=[]\ndef embed_text(texts):\n    calls.append(texts);return np.array([vectors[t] for t in ([texts] if isinstance(texts,str) else texts)])\nself=NS(embed_text=embed_text)\nresult=retrieve(self,"query",[["a","b"],["c","d"]],case+1)',
    wrong=[('(-cosine_scores).argsort()','cosine_scores.argsort()'), ('np.linalg.norm(all_embeddings, axis=1)','np.ones(all_embeddings.shape[0])'), ('self.embed_text(query)[0]','self.embed_text(all_chunks)[1]')],
    hint='문서와 질의를 같은 벡터 공간에서 비교하고 벡터 크기의 영향을 정규화합니다.',
    explanation='top-k 키워드를 묻는 대신 검색기의 데이터 처리 흐름과 유사도 원리를 함께 확인합니다. 문서 처리·임베딩 호출은 제공 객체, 유사도 계산은 실제 NumPy입니다.',mode='CPU 계산 · 문서/임베딩 제공 객체')

add('core-rag-llamaindex','RAG','medium','LlamaIndex: 문서 인덱싱에서 검색까지','관련 Library 활용',RAG+'2일차/code/2. Task_1.ipynb',20,
    span=('base_index =','retrieved_nodes ='),signature='self, documents, query, topk',returns='retrieved_nodes',
    prompt='Document 목록을 제공된 self.parser로 분할·인덱싱하고, 요청한 개수의 관련 노드를 검색하는 LlamaIndex 흐름을 구현하세요. VectorStoreIndex.from_documents, index.as_retriever(similarity_top_k=...), retriever.retrieve가 제공 인터페이스입니다. 인자 이름 자체는 제시하며, 올바른 데이터 연결을 평가합니다.',
    fixture='calls=[]\nclass VectorStoreIndex:\n    @staticmethod\n    def from_documents(documents,transformations):\n        calls.append(("index",documents,transformations))\n        def as_retriever(**kwargs):\n            calls.append(("retriever",kwargs))\n            return NS(retrieve=lambda q:(q,documents[:kwargs["similarity_top_k"]]))\n        return NS(as_retriever=as_retriever)\nresult=(solve(NS(parser="chunk-parser"),["doc-a","doc-b","doc-c"],"question",case+1),calls)',
    wrong=[('transformations=[self.parser]','transformations=[]'), ('documents=documents','documents=[]'), ('base_retriever.retrieve(query)','base_retriever.retrieve("")')],
    hint='같은 문서·parser로 만든 index에서 retriever를 생성하고 원래 질문을 전달합니다.',
    explanation='라이브러리의 생성→설정→검색 단계를 연결하는 핵심 사용법을 검증합니다. 외부 임베딩/API를 호출하지 않는 인터페이스 대체 객체를 사용합니다.',mode='호출 흐름 실행 · LlamaIndex 대체 객체')

add('core-rag-pipeline','RAG','easy','기본 RAG: 검색 근거를 Reader 응답에 연결','RAG 구성 이해 및 활용',RAG+'2일차/code/2. Task_1.ipynb',33,
    symbol='RAG.inference',
    prompt='질문과 검색 자료가 주어졌을 때 Retriever로 근거를 찾고 Reader가 그 근거를 사용해 답하도록 RAG의 전체 inference를 구현하세요. 답변과 사용한 근거를 함께 반환합니다. 두 구성 요소의 인터페이스는 제공 코드의 fixture에서 확인할 수 있습니다.',
    fixture='self=NS(retriever=NS(retrieve=lambda q,docs,k:docs[:k]),reader=NS(generate_response=lambda q,refs:{"question":q,"evidence":refs}))\nresult=inference(self,"q",["a","b","c"],case+1)',
    wrong=[('self.reader.generate_response(query, retrieved_results)','self.reader.generate_response(query, search_results)'), ('self.retriever.retrieve(query, search_results, topk)','self.retriever.retrieve(query, [], topk)'), ('return (answer, retrieved_results)','return (retrieved_results, answer)')],
    hint='Reader가 전체 원본 자료 대신 Retriever가 선택한 근거로 답해야 합니다.',
    explanation='RAG의 핵심인 검색-생성 연결과 근거 추적을 평가합니다. 외부 서비스 없이 작은 대체 구성 요소로 연결을 검증합니다.',mode='호출 흐름 실행 · Retriever/Reader 대체 객체')

add('core-rag-hybrid','RAG','medium','Graph RAG: 질의에 맞는 근거 선택과 생성','RAG 구성 이해 및 활용',RAG+'2일차/code/3. Task_2.ipynb',42,
    symbol='RAGWithSRKG.inference',
    prompt='노트북의 웹 검색+KG RAG 흐름을 구현하세요. 웹 Retriever 결과와 KG 질의 결과를 얻은 뒤, 금융 질의(is_finance=True)에는 KG 결과를 하나의 근거로 사용하고 나머지는 검색 청크를 사용합니다. 선택한 근거로 Reader의 답변을 생성하고 근거도 반환하세요. 이 분기 정책은 원본 실습의 조건입니다.',
    fixture='self=NS(retriever=NS(retrieve=lambda q,docs,k:docs[:k]),kg_query_engine=NS(query=lambda q:("kg-evidence",case!=1)),reader=NS(generate_response=lambda q,refs:{"question":q,"evidence":refs}))\nresult=inference(self,"q",["web-a","web-b","web-c"],case+1)',
    wrong=[('if is_finance:','if not is_finance:'), ('combined_results = [kg_results]','combined_results = retrieved_results'), ('self.reader.generate_response(query, combined_results)','self.reader.generate_response(query, retrieved_results)')],
    hint='자료 출처 선택도 RAG 흐름의 일부입니다. 선택한 동일 근거를 응답과 반환 결과에 반영합니다.',
    explanation='특정 API 응답 문자열을 다루는 대신 질문 유형에 맞는 근거 선택과 생성 단계 연결을 평가합니다.',mode='호출 흐름 실행 · KG/Retriever/Reader 대체 객체')
from quiz_app.expansion import register
register(add, extract, LLM, VIS, OD, DATA, RAG)

from quiz_app.summary_expansion import register as register_summary_expansion
register_summary_expansion(add, extract, LLM, VIS, OD, DATA, RAG)

from quiz_app.coverage_expansion import register as register_coverage
register_coverage(add, extract, normalized, LLM, VIS, OD, DATA, RAG)

# llm_summary는 instruction following을 제외하며, vision_summary에는 별도의
# 학습/평가 루프가 없다. 문제은행도 다섯 summary에 실제로 적힌 범위만 남긴다.
_OUTSIDE_SUMMARY = {
    'core-llm-classification-loss', 'core-llm-collate', 'core-llm-dpo',
    'core-vis-evaluation', 'core-vis-training', 'core-vis-validation',
    'core-rag-verification', 'core-od-activation', 'core-data-scaling',
    'core-data-multistep', 'core-data-training',
}
BANK[:] = [q for q in BANK if q['id'] not in _OUTSIDE_SUMMARY]

BANK.sort(key=lambda q: (FIELDS.index(q['field']), {'easy': 0, 'medium': 1, 'hard': 2}[q['level']]))

BY_ID = {q['id']: q for q in BANK}
assert len(BY_ID) == len(BANK)

def assemble(q, answer):
    answer = textwrap.dedent(answer).strip()
    return q['template'].replace(' ' * q['indent'] + MARKER, textwrap.indent(answer, ' ' * q['indent']))

from quiz_app.mcq import attach
attach(BANK, assemble, normalized, MARKER)
