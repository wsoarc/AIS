"""Summary coverage through coherent notebook operations, not API trivia."""


def register(add, extract, normalized, LLM, VIS, OD, DATA, RAG):
    import ast
    from quiz_app.mcq import entry

    def q(id, field, level, title, path, index, focus, alternatives, fixture,
          prompt, *, symbol=None, span=None, signature=None, returns=None,
          setup='', prefix=None):
        focus = normalized(focus)
        alternatives = [normalized(a) for a in alternatives]
        hide = None
        if symbol and isinstance(ast.parse(extract(path,index,symbol)).body[0], ast.ClassDef):
            hide = extract(path,index,symbol+'.__init__')
        add('core-' + id, field, level, title, 'Summary · ' + title,
            path, index, symbol=symbol, span=span, signature=signature,
            returns=returns, setup=setup, prefix=prefix, hide=hide,
            prompt=prompt, fixture=fixture, wrong=[(focus, a) for a in alternatives],
            hint=prompt, explanation=prompt)
        entry(id, focus, prompt, alternatives)

    q('llm-token-roundtrip', 'LLM', 'easy', 'Tokenizer: 문자열과 토큰 ID의 왕복',
      LLM+'Chapter_2_Exercise_Dataset.ipynb', 0,
      'strings = tokenizer.decode(integers)',
      ['strings = tokenizer.decode(integers[::-1])','strings = tokenizer.decode(integers[:-1])','strings = text + text'],
      'tokenizer=NS(encode=lambda t,**kw:[ord(c) for c in t],decode=lambda ids:"".join(chr(i) for i in ids))\nresult=solve(tokenizer,"abc"+str(case))',
      '제공된 tokenizer로 문자열을 토큰 ID로 인코딩한 뒤 같은 ID 순서를 디코딩해 복원하세요. API 이름은 제공하며 데이터 방향과 순서를 평가합니다.',
      span=('integers =','strings ='),signature='tokenizer, text',returns='integers, strings')

    q('llm-embedding-build', 'LLM', 'medium', 'GPT 구성: 토큰·위치 Embedding과 출력 Head',
      LLM+'Chapter_4_Excercise_GPT.ipynb', 3,
      'self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])',
      ['self.tok_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])','self.tok_emb = nn.Embedding(cfg["emb_dim"], cfg["vocab_size"])','self.tok_emb = nn.Linear(cfg["vocab_size"], cfg["emb_dim"])'],
      'cfg=dict(vocab_size=11,emb_dim=4,context_length=6,drop_rate=0.,n_layers=2)\nm=GPTModel(cfg)\nresult=(m(torch.tensor([[8,9,10]])),tuple(m.pos_emb.weight.shape))',
      '토큰 ID는 어휘 수만큼의 Embedding에서 D차원 표현을 찾습니다. 위치 Embedding과 더해 Transformer를 통과시킨 뒤 V차원 logits으로 투영하세요.',
      symbol='GPTModel',setup='TransformerBlock=lambda cfg:nn.Identity()\nLayerNorm=nn.LayerNorm')

    q('llm-token-loss','LLM','medium','언어 모델: 모든 토큰의 정답과 손실 연결',
      LLM+'Chapter_5_Excercise_Pretraining.ipynb',1,
      'loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())',
      ['loss = torch.nn.functional.cross_entropy(logits[:, -1], target_batch[:, -1])','loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), input_batch.flatten())','loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).softmax(-1), target_batch.flatten())'],
      'result=calc_loss_batch(torch.tensor([[0,1,2],[2,3,4]]),torch.tensor([[1,2,3],[3,4,0]]),nn.Embedding(5,5),"cpu")',
      '(B,T,V) logits의 배치·시간 축을 합치고 (B,T) 정답도 같은 순서로 펼쳐 모든 위치의 next-token cross entropy를 계산하세요.',symbol='calc_loss_batch')

    lora=LLM+'Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb'
    q('llm-lora-init','LLM','medium','LoRA 초기화: 저랭크 경로와 초기 출력 보존',lora,2,
      'self.B = nn.Parameter(torch.zeros(rank, out_dim))',
      ['self.B = nn.Parameter(torch.ones(rank, out_dim))','self.B = nn.Parameter(torch.zeros(out_dim, rank))','self.B = torch.zeros(rank, out_dim)'],
      'm=LoRALayer(4,3,2,4)\nx=torch.randn(2+case,4)\ny=m(x);y.sum().backward()\nresult=(y,[(n,p.grad) for n,p in m.named_parameters()])',
      'A는 (입력,rank), B는 (rank,출력)인 학습 파라미터입니다. B를 0으로 초기화해 최초 adapter 출력은 0이고 이후 gradient로 학습되게 하세요.',symbol='LoRALayer')
    q('llm-lora-freeze','LLM','medium','LoRA 적용: 원본 동결과 Adapter 학습',lora,3,
      'param.requires_grad = False',
      ['param.requires_grad = True','param.data.zero_()','param.detach()'],
      'm=nn.Sequential(nn.Linear(4,3))\nsolve(m,2,4)\nm(torch.randn(2+case,4)).sum().backward()\nresult=[(n,p.requires_grad,p.grad) for n,p in m.named_parameters()]',
      '원본 파라미터를 동결한 다음 LoRA를 주입하여 새 adapter만 학습 가능하게 하세요. 원본 가중치의 값은 보존합니다.',
      span=('for param in model.parameters():','replace_linear_with_lora(model,'),signature='model, LORA_RANK, LORA_ALPHA',
      setup='\n'.join(extract(lora,2,s) for s in ['LoRALayer','LinearWithLoRA','replace_linear_with_lora']))

    q('vis-dataset','Vision','easy','CIFAR10: 학습·평가 분할과 전처리 연결',VIS+'01_ResNet18_CIFAR10.ipynb',5,
      'test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)',
      ['test_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=test_tfms)','test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=train_tfms)','test_set = train_set'],
      'torchvision=NS(datasets=NS(CIFAR10=lambda **kw:kw))\nresult=solve("data","augment","normalize")',
      '학습 데이터에는 학습용 증강을, 평가 데이터에는 평가용 정규화를 연결하세요. 데이터 다운로드는 대체 객체로 처리합니다.',
      span=('train_set =','test_set '),signature='data_root, train_tfms, test_tfms',returns='train_set, test_set')

    q('vis-patch-build','Vision','medium','ViT 구성: 이미지 패치와 분류 토큰',VIS+'02_ViT_CIFAR10.ipynb',10,
      'self.to_patch_embedding = nn.Sequential(Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width), nn.Linear(patch_dim, cfg.dim))',
      ['self.to_patch_embedding = nn.Sequential(Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width), nn.Linear(num_patches, cfg.dim))','self.to_patch_embedding = nn.Sequential(Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width), nn.Linear(cfg.dim, patch_dim))','self.to_patch_embedding = nn.Identity()'],
      'cfg=NS(image_size=8,patch_size=2,channels=3,dim=5,depth=2,heads=1,dim_head=5,mlp_dim=10,pool="cls",dropout=0.,emb_dropout=0.,num_classes=3)\nm=ViT(cfg)\nresult=(m.to_patch_embedding(torch.randn(2,3,8,8)),m.cls_token,m.pos_embedding,m.mlp_head(torch.randn(2,5)))',
      '각 패치의 픽셀·채널을 펼친 차원을 Transformer 차원으로 투영하고, CLS와 위치 Embedding 및 분류 Head를 구성하세요.',symbol='ViT',
      setup='''def pair(x): return (x,x) if isinstance(x,int) else x
class Rearrange(nn.Module):
    def __init__(self, pattern, p1, p2):
        super().__init__(); self.p1=p1; self.p2=p2
    def forward(self,x):
        b,c,h,w=x.shape; p,q=self.p1,self.p2
        return x.reshape(b,c,h//p,p,w//q,q).permute(0,2,4,3,5,1).reshape(b,(h//p)*(w//q),p*q*c)
Transformer=lambda **kw:nn.Identity()''')

    q('vis-doubleconv','Vision','medium','U-Net 구성: 해상도를 보존하는 두 Convolution',VIS+'04_Unet.ipynb',6,
      'self.net = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))',
      ['self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.ReLU())','self.net = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, bias=False), nn.ReLU(), nn.Conv2d(mid_channels, out_channels, 3, bias=False), nn.ReLU())','self.net = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False), nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False))'],
      'm=DoubleConv(2,4,3)\nresult=m(torch.randn(2,2,8+case,8+case))',
      '3×3 Conv→ReLU를 두 번 연결해 입력→중간→출력 채널로 바꾸되 padding=1로 공간 크기를 유지하세요.',symbol='DoubleConv')

    q('vis-unet-build','Vision','hard','U-Net 구성: Encoder·Decoder 채널과 출력 Mask',VIS+'04_Unet.ipynb',6,
      'self.up1 = Up(1024, 512 // factor, bilinear)',
      ['self.up1 = Up(512, 512 // factor, bilinear)','self.up1 = Up(1024, 256 // factor, bilinear)','self.up1 = Up(1024, 512 // factor, not bilinear)'],
      'm=UNet(3,2,bilinear=case==1)\nresult=(m.inc,m.down4,m.up1,m.up2,m.up3,m.up4,m.outc)',
      'Encoder의 채널 확장과 decoder의 skip 결합 채널을 맞추고, bilinear 설정의 factor 및 최종 mask 채널 수를 반영하세요. 블록 생성 기록으로 구성만 검증합니다.',symbol='UNet',
      setup='''def DoubleConv(*a): return ("double",a)
def Down(*a): return ("down",a)
def Up(*a): return ("up",a)
def OutConv(*a): return ("out",a)''')

    adv=OD+'2. Advanced Quantization Lab_answer.ipynb'
    q('od-awq','On-device','hard','AWQ: 중요 채널 보호와 양자화 후 스케일 복원',adv,27,
      'm.weight.data[:, outlier_mask] /= scale_factor',
      ['m.weight.data[:, outlier_mask] *= scale_factor','m.weight.data /= scale_factor','m.weight.data[:, outlier_mask] = 0'],
      'm=nn.Sequential(nn.Linear(100,3,bias=False)); feat={"0":[torch.arange(100.).float()]}\npseudo_quantize_model_weight_scaleup(m,4,-1,feat,2.+case)\nresult=m[0].weight',
      '활성값 중요도로 선택한 입력 채널을 확대해 양자화하고 같은 채널만 원래 스케일로 되돌리세요.',symbol='pseudo_quantize_model_weight_scaleup',setup=extract(adv,17,'pseudo_quantize_tensor'))
    q('od-rotation','On-device','hard','회전 양자화: 입력·출력 Projection의 좌표 변환',adv,67,
      'm.weight.data = R1.T @ W_',
      ['m.weight.data = W_ @ R1','m.weight.data = R1 @ W_','m.weight.data = W_'],
      'm=nn.ModuleDict({"emb":nn.Embedding(7,4),"q_proj":nn.Linear(4,6),"o_proj":nn.Linear(6,4)})\nr=torch.linalg.qr(torch.randn(4,4)).Q\nrotate_model_weight(m,r)\nresult=[p for p in m.parameters()]',
      'Embedding과 입력 Projection에는 오른쪽에 R을, 출력 Projection에는 왼쪽에 R의 전치를 적용하여 weight의 입력·출력 좌표 축을 맞추세요.',symbol='rotate_model_weight')

    kd=OD+'3. Knowledge Distillation_answer.ipynb'
    kd_fixture='''class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.h=nn.Linear(4,5); self.head=nn.Linear(5,3)
    def forward(self,x):
        h=self.h(x); return self.head(h),h
t=Net(); s=Net(); x=torch.randn(3+case,4); labels=torch.arange(3+case)%3
optimizer=torch.optim.Adam(s.parameters(),lr=.01)
loss=solve(t,s,x,labels,.6,.4,optimizer)
result=(loss,[p.grad for p in t.parameters()],[p for p in s.parameters()])'''
    q('od-kd-cosine','On-device','hard','Cosine KD: 숨은 표현과 정답으로 Student 학습',kd,39,
      'loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss',
      ['loss = hidden_rep_loss','loss = label_loss','loss = hidden_rep_loss_weight * hidden_rep_loss - ce_loss_weight * label_loss'],kd_fixture,
      'Teacher의 hidden 표현을 고정하고 Student와의 cosine 손실 및 정답 cross entropy를 가중합해 Student를 갱신하세요. 제공되는 torch.ones wrapper는 원본의 CUDA 이동만 CPU로 대체합니다.',
      span=('with torch.no_grad():','optimizer.step()'),signature='teacher, student, inputs, labels, hidden_rep_loss_weight, ce_loss_weight, optimizer',returns='loss',
      setup='''ce_loss=nn.CrossEntropyLoss(); cosine_loss=nn.CosineEmbeddingLoss()
class CPUOnes:
    def __init__(self,t): self.t=t
    def cuda(self): return self.t
_torch=torch
torch=NS(no_grad=_torch.no_grad,ones=lambda *a,**kw:CPUOnes(_torch.ones(*a,**kw)),**{n:getattr(_torch,n) for n in ['randn','arange','optim']})''')
    q('od-kd-hint','On-device','hard','Hint KD: Feature Map 회귀와 정답 학습',kd,52,
      'hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)',
      ['hidden_rep_loss = mse_loss(regressor_feature_map, regressor_feature_map.detach())','hidden_rep_loss = mse_loss(student_logits, teacher_feature_map)','hidden_rep_loss = mse_loss(regressor_feature_map, -teacher_feature_map)'],kd_fixture,
      'Student의 regressor feature map을 Teacher feature map에 MSE로 맞추고 label loss와 결합하세요. Teacher gradient는 차단합니다.',
      span=('with torch.no_grad():','optimizer.step()'),signature='teacher, student, inputs, labels, feature_map_weight, ce_loss_weight, optimizer',returns='loss',setup='ce_loss=nn.CrossEntropyLoss(); mse_loss=nn.MSELoss()')

    kv=OD+'4_[Colab]_KV_Cache_Pruning_answer.ipynb'
    q('od-cache-memory','On-device','easy','KV Cache: 메모리 사용량과 GQA Head 수',kv,9,
      'total_bytes = 2 * batch_size * n_layers * n_kv_heads * seq_len * head_dim * dtype_bytes',
      ['total_bytes = batch_size * n_layers * n_kv_heads * seq_len * head_dim * dtype_bytes','total_bytes = 2 * batch_size * n_layers * n_q_heads * seq_len * head_dim * dtype_bytes','total_bytes = 2 * batch_size * n_layers * n_kv_heads * seq_len ** 2 * head_dim * dtype_bytes'],
      'config=NS(num_hidden_layers=4,num_attention_heads=8,num_key_value_heads=2,hidden_size=64);model=NS(config=config)\nresult=kv_cache_bytes(config,16+case,2,2)',
      'Key와 Value 각각의 (배치,KV head,문맥 길이,head 차원) 저장량을 모든 레이어에 합산하세요. GQA에서는 query head 수와 KV head 수가 다릅니다.',symbol='kv_cache_bytes')
    q('od-streaming','On-device','medium','StreamingLLM: Sink와 최근 토큰 보존',kv,17,
      'scores[:, :, :n_sink] = float("inf")',
      ['scores[:, :, -n_sink:] = float("inf")','scores[:, :, :n_sink] = 0','scores[:, :, :] = float("inf")'],
      'layer=NS(keys=torch.randn(2,2,8,3))\nresult=streaming_score(None,layer,5,n_sink=2+case)',
      '위치가 최근일수록 점수를 높이고 맨 앞 sink token에 무한대 점수를 주어 두 종류의 토큰을 함께 보존하세요.',symbol='streaming_score',setup='DEVICE="cpu"')
    q('od-h2o-norm','On-device','medium','H2O-Norm: 누적 Attention의 위치 편향 보정',kv,23,
      'return scores / count.view(1, 1, K)',
      ['return scores','return scores * count.view(1, 1, K)','return scores / K'],
      'result=h2o_norm_score(torch.rand(2,2,6,6),None,2+case)',
      '각 key가 attention을 받을 수 있었던 query 수로 누적 점수를 나누고 최근 토큰 보호를 유지하세요.',symbol='h2o_norm_score')
    q('od-snapkv','On-device','hard','SnapKV: 관측 Window와 이웃 중요도 결합',kv,26,
      'scores = attn_kv[:, :, -w:, :].sum(dim=2)',
      ['scores = attn_kv[:, :, :w, :].sum(dim=2)','scores = attn_kv.sum(dim=2)','scores = attn_kv[:, :, -w:, :].sum(dim=1)'],
      'result=snapkv_score(torch.rand(2,2,8,8),None,5,window=2+case,kernel=3)',
      '마지막 관측 window의 query attention을 누적하고 주변 토큰의 최대 점수를 pooling한 뒤 최근 window를 보호하세요.',symbol='snapkv_score')

    spec=OD+'5_[Colab]Speculative_Decoding_Answer.ipynb'
    q('od-draft','On-device','medium','Speculative Draft: 토큰과 Cache의 연속 갱신',spec,7,
      'x, cache = greedy_step(model, x, cache)',
      ['x, cache = greedy_step(model, x, None)','x, cache = greedy_step(model, x * 0, cache)','x, cache = greedy_step(model, x, 0)'],
      'def greedy_step(m,x,c):return x+1,(c or 0)+1\nresult=draft_tokens(None,torch.tensor([[2]]),4,2+case)',
      'Draft 모델이 방금 생성한 토큰과 갱신한 cache를 다음 step에 전달하여 gamma개 후보를 순차적으로 생성하세요.',symbol='draft_tokens')
    q('od-verify','On-device','medium','Target 검증: 확정 토큰과 Draft의 위치 대응',spec,7,
      'verify_ids = torch.cat([x, drafted], dim=-1)',
      ['verify_ids = torch.cat([drafted, x], dim=-1)','verify_ids = drafted','verify_ids = torch.cat([x, drafted], dim=0)'],
      'def target(ids,**kw):return NS(logits=F.one_hot((ids+1)%9,9).float(),past_key_values=kw["past_key_values"]+1)\nresult=verify_draft(target,torch.tensor([[1]]),torch.tensor([[2,3,4]]),case)',
      '마지막 확정 토큰을 draft 앞에 붙여 Target이 각 후보와 bonus token을 예측하도록 위치를 맞추고 cache를 전달하세요.',symbol='verify_draft')
    q('od-dflash','On-device','hard','DFlash: Hidden Block에서 병렬 후보 생성',spec,29,
      'draft_tokens = torch.argmax(draft_logits, dim=-1)',
      ['draft_tokens = torch.argmax(draft_logits, dim=1)','draft_tokens = torch.argmax(draft_logits[:, -1], dim=-1)','draft_tokens = torch.argmin(draft_logits, dim=-1)'],
      'LAST_DRAFT=None\nclass Drafter:\n    def __call__(self,**kw):return kw["target_hidden"]\n    def compute_logits(self,h,head):return head(h)\nresult=parallel_block_draft(Drafter(),torch.randn(2,7,4),None,None,None,nn.Linear(4,9),3+case)',
      '마지막 verify_size−1개 hidden state에 출력 head를 적용하고 각 위치의 어휘 축에서 토큰을 골라 한 번에 draft block을 만드세요.',symbol='parallel_block_draft')

    ts=DATA+'3_4-ts-practice/ts_solution.ipynb'
    q('data-time-split','Data','easy','시계열 분할: 과거 학습과 미래 평가',ts,11,
      'test_data = df[training_data_len:][["Open"]]',
      ['test_data = df[:training_data_len][["Open"]]','test_data = df[training_data_len + 1:][["Open"]]','test_data = df[["Open"]]'],
      'df=pd.DataFrame({"Open":np.arange(11+case)})\na,b=solve(df)\nresult=(a.values,b.values)',
      '시간 순서대로 앞 80%를 학습, 나머지 미래 구간을 평가에 사용하세요. 두 구간은 겹치지 않아야 합니다.',span=('train_ratio =','test_data ='),signature='df',returns='train_data, test_data',setup='import pandas as pd')
    q('data-rnn','Data','medium','RNN: 시점별 Hidden과 예측 연결',ts,34,
      'self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)',
      ['self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)','self.rnn = nn.RNN(hidden_size, input_size, num_layers, batch_first=True)','self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)'],
      'm=RNNModel(2,4,2)\nresult=m(torch.randn(3,5+case,2))',
      'batch_first RNN의 모든 시점 hidden을 Linear에 전달해 (B,T,1)을 반환하세요. 마지막 hidden state만 사용하지 않습니다.',symbol='RNNModel')
    ncf=DATA+'7_8-recsys-practice/RecSys_NCF.ipynb'
    q('data-recsys-split','Data','medium','추천 데이터: 연속 ID와 평점 분포 유지',ncf,8,
      'df_train, df_test = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.rating.values)',
      ['df_train, df_test = model_selection.train_test_split(df, test_size=0.1, random_state=42)','df_train, df_test = model_selection.train_test_split(df, test_size=0.9, random_state=42, stratify=df.rating.values)','df_train, df_test = df, df'],
      'df=pd.DataFrame({"userId":np.tile([10,30],50),"movieId":np.arange(100)*3,"rating":np.tile([1.,2.,3.,4.,5.],20)})\na,b=solve(df)\nresult=(a.values,b.values)',
      '사용자·아이템 ID를 연속 정수로 인코딩하고 평점별 분포를 유지하여 train/test를 분할하세요.',span=('lbl_user =','df_train, df_test ='),signature='df',returns='df_train, df_test',setup='import pandas as pd\nfrom sklearn import preprocessing, model_selection')
    q('data-ncf-train','Data','medium','NCF 학습: 평점 Shape와 회귀 손실',ncf,14,
      'ground_truth = train_data["ratings"].view(batch_size, -1).to(torch.float32)',
      ['ground_truth = train_data["ratings"].to(torch.float32)','ground_truth = prediction.detach()','ground_truth = train_data["users"].view(batch_size, -1).to(torch.float32)'],
      'model=Neural_Collaborative_Filtering(5,6);optimizer=torch.optim.Adam(model.parameters(),lr=.01)\nd=dict(users=torch.tensor([0,1,2]),movies=torch.tensor([2,3,4]),ratings=torch.tensor([1.,2.5,4.]))\nloss=solve(model,d,3,nn.MSELoss(),optimizer,0.)\nresult=(loss,[p for p in model.parameters()])',
      'NCF의 (B,1) 평점 예측과 같은 shape의 float 정답으로 MSE를 계산하고 역전파·갱신하세요. 의도치 않은 broadcasting을 방지합니다.',span=('prediction = model','optimizer.step()'),signature='model, train_data, batch_size, loss_func, optimizer, total_loss',returns='loss',setup=extract(ncf,10,'Neural_Collaborative_Filtering'))
    gcf=DATA+'7_8-recsys-practice/RecSys_GCF_sol.ipynb'
    q('data-graph-edges','Data','medium','추천 Graph: 사용자와 아이템 Node 분리',gcf,9,
      'dst.append(row["movieId"] + num_users)',
      ['dst.append(row["movieId"])','dst.append(row["userId"] + num_users)','dst.append(row["movieId"] + num_movies)'],
      'num_users=3;num_movies=5\ndf=pd.DataFrame({"userId":[0,1,2],"movieId":[1,3,4],"rating":[.5,3.,4.]})\nresult=create_edge_index(df)',
      '평점 threshold를 만족한 상호작용만 edge로 만들고, 아이템 ID에 사용자 수를 더해 서로 다른 node 공간에 배치하세요.',symbol='create_edge_index',setup='import pandas as pd')
    q('data-graph-split','Data','medium','추천 Graph: Train·Validation·Test Edge 분리',gcf,10,
      'val_indices, test_indices = train_test_split(test_indices, test_size=0.5)',
      ['val_indices, test_indices = train_test_split(train_indices, test_size=0.5)','val_indices, test_indices = test_indices, test_indices','val_indices, test_indices = train_test_split(test_indices, test_size=0.2)'],
      'result=solve(torch.arange(100).reshape(2,50))',
      '전체 edge의 80%를 학습에, 남은 20%를 validation과 test로 다시 나누어 겹치지 않는 세 집합을 만드세요.',span=('train_indices,','test_edge_index ='),signature='edge_index',returns='train_edge_index, val_edge_index, test_edge_index',setup='from sklearn.model_selection import train_test_split')
    q('data-ngcf-representation','Data','hard','NGCF: 여러 깊이의 표현과 사용자·아이템 분리',gcf,14,
      'final_features = torch.concat(layer_outputs, dim=-1)',
      ['final_features = torch.stack(layer_outputs).mean(0)','final_features = torch.concat(layer_outputs, dim=0)','final_features = node_features'],
      'self=NS(node_embeddings=nn.Embedding(7,3),layers=[lambda e,x,u,i:x*2,lambda e,x,u,i:x+.5],num_users=3,num_items=4)\nresult=forward(self,torch.tensor([[0,1],[3,4]]))',
      '초기 embedding과 각 graph layer 표현을 feature 축으로 이어 붙이고 사용자·아이템 node 구간을 나누세요.',symbol='NGCF.forward')
    q('data-bpr-train','Data','hard','NGCF 학습: 사용자·Positive·Negative와 BPR 연결',gcf,18,
      'neg_emb = item_features[neg_item_indices]',
      ['neg_emb = item_features[pos_item_indices]','neg_emb = user_features[neg_item_indices]','neg_emb = item_features[neg_item_indices].detach()'],
      'class Model(nn.Module):\n    bpr_loss=bpr_loss\n    def __init__(self):\n        super().__init__();self.u=nn.Parameter(torch.randn(3,4));self.i=nn.Parameter(torch.randn(6,4))\n    def forward(self,e):return self.u,self.i\nm=Model();opt=torch.optim.Adam(m.parameters(),lr=.01)\nloss=solve(m,None,torch.tensor([0,1]),torch.tensor([1,2]),torch.tensor([4,5]),opt)\nresult=(loss,m.u,m.i)',
      '동일한 사용자의 positive·negative 아이템 표현을 별도로 선택하여 BPR 손실을 계산하고 graph embedding을 갱신하세요.',span=('user_features, item_features = model','optimizer.step()'),signature='model, train_edge_index, user_indices, pos_item_indices, neg_item_indices, optimizer',returns='loss',setup=extract(gcf,14,'NGCF.bpr_loss'))

    task=RAG+'2일차/code/2. Task_1.ipynb'
    intro=RAG+'1일차/code/1. Llama_index.ipynb'
    q('rag-load-chunk','RAG','medium','RAG 준비: 문서 로딩에서 겹치는 Chunk까지',intro,20,
      'nodes = parser.get_nodes_from_documents(documents)',
      ['nodes = parser.get_nodes_from_documents([])','nodes = documents','nodes = parser.get_nodes_from_documents(documents[::-1])'],
      'SimpleDirectoryReader=lambda p:NS(load_data=lambda:["doc-a","doc-b"])\nSentenceSplitter=lambda **kw:NS(get_nodes_from_documents=lambda ds:[(d,kw) for d in ds])\nresult=solve()',
      'Reader가 읽은 Document를 지정된 chunk_size와 overlap의 splitter로 분할하세요. 설정값은 제공하며 문서→node 흐름을 평가합니다.',span=('parser =','nodes ='),signature='',returns='nodes',prefix=dict(path=intro,index=11,span=('documents =','documents =')))
    q('rag-embedding-call','RAG','medium','Embedding: 단일 질의와 문서 배치의 공통 벡터 공간',task,16,
      'response = self.client.embeddings.create(model="text-embedding-3-small", input=texts)',
      ['response = self.client.embeddings.create(model="text-embedding-3-small", input=texts[::-1])','response = self.client.embeddings.create(model="text-embedding-3-small", input=texts[:1])','response = self.client.embeddings.create(model="text-embedding-3-small", input=[])'],
      'self=NS(client=NS(embeddings=NS(create=lambda **kw:NS(data=[NS(embedding=[len(t),ord(t[0])]) for t in kw["input"]]))))\nresult=(embed_text(self,"query"),embed_text(self,["a","bb","ccc"]))',
      '단일 문자열은 길이 1의 배치로 바꾸고 문서 목록은 순서를 유지해 같은 embedding 모델에 전달하세요. 응답 벡터를 (N,D) 배열로 구성합니다.',symbol='BaseRetriever.embed_text')
    q('rag-html','RAG','medium','검색 문서 전처리: HTML에서 본문 추출',RAG+'2일차/code/3. Task_2.ipynb',41,
      'text = soup.get_text(" ", strip=True)',
      ['text = html_text["page_snippet"]','text = html_text["page_result"]','text = ""'],
      'result=parse_htmls([dict(page_result="<p>alpha</p><p>beta</p>",page_snippet="short")])',
      '검색 결과의 page_result HTML에서 태그를 제거하고 본문을 공백으로 연결해 chunking에 넘길 문서 목록을 만드세요. parser는 제공됩니다.',symbol='parse_htmls',
      setup='''from html.parser import HTMLParser
class BeautifulSoup(HTMLParser):
    def __init__(self,text,features):
        super().__init__();self.parts=[];self.feed(text)
    def handle_data(self,text): self.parts.append(text)
    def get_text(self,sep,strip): return sep.join(t.strip() for t in self.parts if t.strip())''')
    q('rag-reader-generate','RAG','medium','Reader 생성: 근거 Prompt에서 답변 추출',task,29,
      'llm_input = self.prompt_generator(query, top_k_chunks)',
      ['llm_input = self.prompt_generator(query, [])','llm_input = self.prompt_generator(top_k_chunks, query)','llm_input = []'],
      'self=NS(prompt_generator=lambda q,refs:[q,refs])\ndef create(**kw):return NS(choices=[NS(message=NS(content=repr(kw["messages"])))])\noai_client=NS(chat=NS(completions=NS(create=create)))\nresult=generate_response(self,"q",["evidence-"+str(case)])',
      '질문과 검색 근거로 만든 prompt를 생성 모델에 전달하고 첫 답변의 content를 반환하세요. 외부 API는 호출 기록용 객체로 대체합니다.',symbol='Reader.generate_response')
    mcp=RAG+'2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb'
    q('rag-mcp-discovery','RAG','medium','MCP 연결: Client에서 Tool·Resource 조회까지',mcp,66,
      'tools = await mcp_tool.to_tool_list_async()',
      ['tools = mcp_tool.to_tool_list_async()','tools = []','tools = [mcp_tool]'],
      'BasicMCPClient=lambda url:NS(url=url)\nclass McpToolSpec:\n    def __init__(self,client):self.client=client\n    async def to_tool_list_async(self):return [self.client.url,"tool"]\n    async def fetch_resources(self):return [self.client.url,"resource"]\nresult=asyncio.run(solve("local-sse"))',
      'MCP endpoint의 Client를 ToolSpec으로 감싸고 비동기 tool 변환 결과를 받아 Agent가 사용할 목록을 구성하세요.',
      span=('resources =','resources ='),signature='external_mcp_server',returns='tools, resources',
      prefix=dict(path=mcp,index=65,span=('mcp_client =','tools = await')))

    q('od-awq-scales','On-device','hard','AWQ: 활성값 기반 채널별 Scale 적용',adv,34,
      'scales = s_x ** ratio',
      ['scales = s_x * ratio','scales = s_x ** (1 - ratio)','scales = torch.ones_like(s_x)'],
      'fc=nn.Linear(4,3,bias=False)\nsolve(torch.tensor([.2,1.,3.,8.]),.2+case*.2,[fc],4,-1)\nresult=fc.weight',
      '평균 활성값의 ratio 거듭제곱으로 채널별 scale을 만들고 가중치를 확대→양자화→축소하세요. ratio와 정규화 식은 제공됩니다.',
      span=('scales = s_x ** ratio','fc.weight.data /= scales'),signature='s_x, ratio, linears2scale, w_bit, q_group_size',setup=extract(adv,17,'pseudo_quantize_tensor'))

    q('rag-grade','RAG','medium','RAG 평가: 정확한 답·허용 답·보류·오답 구분',mcp,36,
      'if CRAG_evaluation(item["query"], ground_truth, prediction) == 1:\n    return "acceptable"',
      ['if CRAG_evaluation(item["query"], ground_truth, prediction) == 0:\n    return "acceptable"','if CRAG_evaluation(item["query"], ground_truth, prediction) == 1:\n    return "perfect"','if CRAG_evaluation(item["query"], ground_truth, prediction) == 1:\n    return "missing"'],
      'CRAG_evaluation=lambda q,g,p:int(p=="accepted")\nitem={"query":"q","answer":"exact"}\nresult=[grade_prediction(item,p) for p in ["exact","accepted","I don\'t know","wrong"]]',
      '정규화한 답변의 정확 일치와 답변 보류를 먼저 구분한 뒤 judge가 허용한 답과 오답을 나누세요.',symbol='grade_prediction')

    q('data-mape','Data','easy','시계열 평가: 절대 오차와 상대 오차',ts,27,
      'mape = mean_absolute_percentage_error(y_test, test_predictions)',
      ['mape = mean_absolute_percentage_error(test_predictions, y_test)','mape = mean_squared_error(y_test, test_predictions)','mape = mean_absolute_percentage_error(y_test, test_predictions) * 100'],
      'model=nn.Linear(1,1)\nresult=test(model,torch.arange(1.,13.).reshape(3,4,1),torch.tensor([2.,5.,9.]))',
      'RMSE와 함께 정답값 기준 상대 오차인 sklearn MAPE를 반환하세요. sklearn 출력은 백분율 숫자가 아니라 비율입니다.',symbol='test',setup='device="cpu"\nfrom sklearn.metrics import mean_squared_error, mean_absolute_percentage_error')
    q('data-ndcg','Data','hard','추천 평가: 순위 할인과 이상적인 DCG',gcf,16,
      'dcg += 1.0 / math.log2(rank + 2)',
      ['dcg += 1.0','dcg += 1.0 / math.log2(rank + 1)','dcg += math.log2(rank + 2)'],
      'num_users=2\nu=torch.tensor([[1.,0.],[0.,1.]])\ni=torch.tensor([[1.,0.],[.5,.5],[0.,1.]])\ne=torch.tensor([[0,1],[4,2]])\nresult=evaluate(u,i,e,2+case%2)',
      '관련 아이템이 높은 순위에 있으면 더 큰 DCG 기여도를 부여하고 이상적인 순위의 IDCG로 나눠 NDCG를 구하세요. Recall과 Precision도 함께 반환합니다.',symbol='evaluate',setup='from collections import defaultdict')

    q('rag-mcp-agent','RAG','hard','MCP Agent: 도구·LLM·대화 Context 구성',mcp,69,
      'self.agent_context = Context(self.agent)',
      ['self.agent_context = None','self.agent_context = Context(self.llm)','self.agent_context = Context(self.mcp_tool_spec)'],
      'async def tool_list():return [NS(metadata=NS(description="tool"))]\nself=NS(mcp_tool_spec=NS(to_tool_list_async=tool_list),llm="llm",agent=None,agent_context=None)\nasyncio.run(init_agent(self))\nresult=(self.agent,self.agent_context)',
      '발견한 MCP tools와 LLM으로 FunctionAgent를 생성한 뒤 그 Agent에 연결된 Context를 초기화하세요. 다음 질의는 이 상태를 전달받습니다.',symbol='KGQueryEngineWithMCP.init_agent',
      setup='SYSTEM_PROMPT="Use tools"\nFunctionAgent=lambda **kw:kw\nContext=lambda agent:{"owner":agent}')
    q('rag-mcp-query','RAG','hard','MCP 실행: Context 전달과 비동기 답변 수신',mcp,69,
      'handler = self.agent.run(question, ctx=self.agent_context)',
      ['handler = self.agent.run(question, ctx=None)','handler = self.agent.run("", ctx=self.agent_context)','handler = self.agent.run(self.agent_context, ctx=question)'],
      'class Handler:\n    def __init__(self,q,ctx):self.q=q;self.ctx=ctx\n    async def stream_events(self):\n        if False:yield None\n    def __await__(self):\n        async def finish():return repr((self.q,self.ctx))\n        return finish().__await__()\nself=NS(agent=NS(run=lambda q,ctx:Handler(q,ctx)),agent_context={"turn":case})\nresult=asyncio.run(query(self,"question"))',
      'Agent에 질의와 같은 대화 Context를 전달하고 streaming handler의 완료를 await하여 최종 응답을 반환하세요.',symbol='KGQueryEngineWithMCP.query')

    q('rag-mcp-time','RAG','medium','MCP RAG: 질의 시각을 포함한 검색 요청',mcp,78,
      'mcp_result = await self.mcp_application.query(full_query, verbose=False)',
      ['mcp_result = await self.mcp_application.query(query, verbose=False)','mcp_result = await self.mcp_application.query(query_time, verbose=False)','mcp_result = self.mcp_application.query(full_query, verbose=False)'],
      'calls=[]\nasync def init():calls.append("init")\nasync def query(q,**kw):calls.append(q);return q\nself=NS(mcp_application=NS(init_agent=init,query=query))\nresult=(asyncio.run(retrieve(self,"price now","2026-08-01",[],5)),calls)',
      'Agent를 초기화하고 원래 질문과 query_time을 함께 전달하여 상대적인 시간 표현의 기준을 명시하세요.',symbol='RAGwithMCP.retrieve')
