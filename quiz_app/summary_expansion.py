"""Questions added from topics explicitly listed in the five summary files."""


def register(add, extract, LLM, VIS, OD, DATA, RAG):
    from quiz_app.mcq import entry

    def question(id, field, level, title, point, path, index, symbol, prompt,
                 fixture, focus, alternatives, explanation, setup=''):
        corrections = ()
        if id == 'data-ngcf-layer':
            corrections = [('aggregated_messages.index_add(0, src, edge_messages_for_src)',
                            'aggregated_messages.index_add_(0, src, edge_messages_for_src)')]
            explanation += ' 원본 셀의 사용자 방향 index_add 반환값 누락은 summary와 동일한 index_add_로 교정했습니다.'
        add('core-' + id, field, level, title, point, path, index, symbol=symbol,
            prompt=prompt, fixture=fixture, wrong=[(focus, alt) for alt in alternatives],
            hint=explanation, explanation=explanation, setup=setup, corrections=corrections)
        entry(id, focus, prompt, alternatives)

    question('llm-generation', 'LLM', 'medium', 'Greedy 생성: 마지막 토큰에서 다음 토큰 선택', '모델의 원리 이해',
        LLM + 'Chapter_4_Excercise_GPT.ipynb', 4, 'generate_text_simple',
        '문맥 길이를 제한하면서 모델의 마지막 위치 logits에서 greedy 다음 토큰을 골라 시퀀스에 추가하세요.',
        'model=lambda x:F.one_hot((x+1)%7,7).float()*5\nresult=generate_text_simple(model,torch.tensor([[0,1,2]]),2+case,3)',
        'logits = logits[:, -1, :]',
        ['logits = logits[:, 0, :]', 'logits = logits.mean(dim=1)', 'logits = logits[:, :, -1]'],
        '다음 토큰은 현재 문맥의 마지막 위치가 예측한 어휘 logits에서 선택합니다.')

    question('llm-lora', 'LLM', 'medium', 'LoRA: 저랭크 행렬과 스케일 적용', '모델의 원리 이해',
        LLM + 'Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb', 2, 'LoRALayer.forward',
        '입력에 A와 B를 차례로 적용하고 alpha/rank로 스케일하는 LoRA adapter 출력을 구현하세요.',
        'self=NS(A=torch.randn(4,2),B=torch.randn(2,3),alpha=4.,rank=2)\nresult=forward(self,torch.randn(2+case,4))',
        'x = self.alpha / self.rank * (x @ self.A @ self.B)',
        ['x = self.alpha * (x @ self.A @ self.B)', 'x = self.rank / self.alpha * (x @ self.A @ self.B)', 'x = self.alpha / self.rank * (x @ self.B @ self.A)'],
        'LoRA의 갱신은 입력→A→B 순서이며 alpha/rank 비율을 곱합니다.')

    question('llm-linear-lora', 'LLM', 'easy', 'LoRA Linear: 원본 출력과 Adapter 결합', '모델의 원리 이해',
        LLM + 'Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb', 2, 'LinearWithLoRA.forward',
        '동결 가능한 원본 Linear 출력에 학습 가능한 LoRA 출력을 잔차처럼 더하세요.',
        'self=NS(linear=nn.Linear(4,3),lora=nn.Linear(4,3,bias=False))\nresult=forward(self,torch.randn(2+case,4))',
        'return self.linear(x) + self.lora(x)',
        ['return self.linear(x)', 'return self.lora(x)', 'return self.linear(x) * self.lora(x)'],
        'LoRA는 원본 Linear를 대체하지 않고 저랭크 adapter 출력을 더합니다.')

    question('vis-resnet-cifar', 'Vision', 'easy', 'ResNet18: CIFAR-10 입력과 출력에 맞춘 수정', '모델의 원리 이해',
        VIS + '01_ResNet18_CIFAR10.ipynb', 8, 'build_resnet18_for_cifar10',
        '32×32 CIFAR-10에 맞게 ResNet18의 stem을 3×3 stride 1로 바꾸고 초기 maxpool을 제거하세요.',
        'model=build_resnet18_for_cifar10(7+case)\nx=torch.randn(1,3,32,32)\nh=model.conv1(x)\nresult=(h,model.maxpool(h),model.fc(torch.randn(1,model.fc.in_features)))',
        'model.maxpool = nn.Identity()',
        ['model.maxpool = nn.MaxPool2d(2)', 'model.maxpool = nn.AdaptiveAvgPool2d(1)', 'model.maxpool = nn.Conv2d(64, 64, 1)'],
        '작은 CIFAR 영상에서는 ImageNet용 초기 maxpool을 제거하여 공간 정보를 보존합니다.',
        setup='from torchvision import models')

    question('vis-attention', 'Vision', 'hard', 'ViT Attention: QK 점수와 V 가중합', '모델의 원리 이해',
        VIS + '02_ViT_CIFAR10.ipynb', 8, 'Attention.forward',
        '여러 head의 Q·K 유사도를 스케일링하고 softmax한 뒤 V를 가중합하여 토큰 표현을 만드세요.',
        'self=NS(heads=2,to_qkv=nn.Linear(4,12,bias=False),scale=2**-.5,attend=nn.Softmax(-1),last_attn=None,to_out=nn.Linear(4,4))\nresult=forward(self,torch.randn(2,3+case,4))',
        "dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale",
        ["dots = einsum('b h i d, b h j d -> b h i j', q, v) * self.scale", "dots = einsum('b h i d, b h j d -> b h i j', q, k)", "dots = einsum('b h i d, b h j d -> b h i j', k, v) * self.scale"],
        'Attention score는 Q와 K의 내적을 head 차원의 제곱근으로 스케일링합니다.',
        setup='''from torch import einsum
def rearrange(t, pattern, h=None):
    if h is not None:
        b,n,d=t.shape
        return t.reshape(b,n,h,d//h).transpose(1,2)
    b,h,n,d=t.shape
    return t.transpose(1,2).reshape(b,n,h*d)''')

    question('vis-down', 'Vision', 'easy', 'U-Net Down: 해상도 축소 후 특징 추출', '모델의 원리 이해',
        VIS + '04_Unet.ipynb', 6, 'Down.forward',
        'MaxPool로 공간 해상도를 절반으로 줄인 다음 DoubleConv로 특징을 추출하세요.',
        'self=NS(pool=nn.MaxPool2d(2),conv=nn.Conv2d(2,4,3,padding=1))\nresult=forward(self,torch.randn(2,2,8+case*2,8+case*2))',
        'x = self.pool(x)',
        ['x = self.conv(x)', 'x = F.interpolate(x, scale_factor=2)', 'x = x.mean(dim=1)'],
        'Down block은 pooling 이후 convolution 순서로 공간 크기를 줄이고 채널 특징을 만듭니다.')

    question('od-scale-zero', 'On-device', 'easy', '선형 양자화: Scale과 Zero Point 계산', 'Quantization',
        OD + '1. Basic Quantization Lab_answer.ipynb', 33, 'get_quantization_scale_and_zero_point',
        '실수 최솟값·최댓값을 정수 범위에 선형 대응시키는 scale과 zero point를 계산하세요.',
        'result=get_quantization_scale_and_zero_point(torch.tensor([-2.,1.+case]),4+case)',
        'scale = (fp_max - fp_min) / (quantized_max - quantized_min)',
        ['scale = (fp_max + fp_min) / (quantized_max - quantized_min)', 'scale = (fp_max - fp_min) / quantized_max', 'scale = (quantized_max - quantized_min) / (fp_max - fp_min)'],
        'Scale은 실수 범위의 폭을 정수 코드 범위의 폭으로 나눈 값입니다.',
        setup=extract(OD + '1. Basic Quantization Lab_answer.ipynb', 22, 'get_quantized_range'))

    question('od-kmeans', 'On-device', 'medium', 'K-means 양자화: Codebook으로 Tensor 복원', 'Quantization',
        OD + '1. Basic Quantization Lab_answer.ipynb', 85, 'k_means_quantize',
        '각 원소의 cluster label로 centroid를 조회하여 양자화된 tensor를 구성하세요. codebook은 제공됩니다.',
        'Codebook=lambda c,l:NS(centroids=c,labels=l)\nx=torch.randn(2,3);cb=Codebook(torch.tensor([-1.,.5,2.]),torch.tensor([0,1,2,1,0,2]))\nresult=(k_means_quantize(x,codebook=cb),x)',
        'quantized_tensor = codebook.centroids[codebook.labels]',
        ['quantized_tensor = codebook.labels[codebook.centroids]', 'quantized_tensor = codebook.centroids', 'quantized_tensor = codebook.labels.float()'],
        'Codebook decoding은 각 원소의 label 위치에 해당하는 centroid를 선택합니다.')

    question('od-qat-codebook', 'On-device', 'medium', 'QAT: 학습된 가중치로 Centroid 갱신', 'Quantization',
        OD + '1. Basic Quantization Lab_answer.ipynb', 96, 'update_codebook',
        '고정된 cluster label별로 현재 실수 가중치 평균을 계산해 centroid를 갱신하세요.',
        'cb=NS(centroids=torch.zeros(2),labels=torch.tensor([0,0,1,1]))\nresult=(update_codebook(torch.tensor([1.,3.,4.,8.])+case,cb),cb.centroids)',
        'codebook.centroids[k] = torch.mean(fp32_tensor[codebook.labels == k])',
        ['codebook.centroids[k] = torch.mean(fp32_tensor)', 'codebook.centroids[k] = torch.sum(fp32_tensor[codebook.labels == k])', 'codebook.centroids[k] = fp32_tensor[k]'],
        'QAT의 codebook 갱신은 각 label에 속한 현재 가중치들의 평균을 사용합니다.')

    question('data-ncf', 'Data', 'medium', 'NCF: 사용자·아이템 Embedding 결합', '모델의 원리 이해',
        DATA + '7_8-recsys-practice/RecSys_NCF.ipynb', 10, 'Neural_Collaborative_Filtering.forward',
        '사용자와 영화 embedding을 feature 축으로 연결하고 MLP로 평점을 예측하세요.',
        'self=NS(user_embedding=nn.Embedding(5,32),movie_embedding=nn.Embedding(7,32),fc1=nn.Linear(64,32),relu=nn.ReLU(),fc2=nn.Linear(32,1))\nresult=forward(self,torch.tensor([0,1,2]),torch.tensor([2,3,4]))',
        'input_embedding = torch.cat([user_embedding, movie_embedding], dim=1)',
        ['input_embedding = user_embedding + movie_embedding', 'input_embedding = torch.cat([user_embedding, movie_embedding], dim=0)', 'input_embedding = user_embedding * movie_embedding'],
        'NCF는 두 32차원 embedding을 feature 축으로 이어 64차원 MLP 입력을 만듭니다.')

    question('data-ngcf-layer', 'Data', 'hard', 'NGCF: 이웃 메시지의 Degree 정규화', '모델의 원리 이해',
        DATA + '7_8-recsys-practice/RecSys_GCF_sol.ipynb', 13, 'NGCFLayer.forward',
        '사용자–아이템 edge 메시지를 양 끝 node degree의 제곱근으로 정규화하여 집계하세요.',
        'self=NS(W1=nn.Linear(3,3),W2=nn.Linear(3,3),leaky_relu=nn.LeakyReLU(.2))\nedge=torch.tensor([[0,0,1],[2,3,3]])\nresult=forward(self,edge,torch.randn(4,3),2,2)',
        'norm = 1.0 / torch.sqrt(deg[src] * deg[dst])',
        ['norm = deg[src] * deg[dst]', 'norm = 1.0 / (deg[src] + deg[dst])', 'norm = torch.sqrt(deg[src] * deg[dst])'],
        'NGCF의 대칭 정규화 계수는 1/sqrt(deg(u)deg(i))입니다.')

    question('data-ranking-metrics', 'Data', 'hard', '추천 평가: Recall·Precision·NDCG', '평가 Metric 이해',
        DATA + '7_8-recsys-practice/RecSys_GCF_sol.ipynb', 16, 'evaluate',
        '추천 순위에서 hit 수로 Recall@K와 Precision@K를 계산하고 순위를 반영한 NDCG를 구하세요.',
        'num_users=2\nu=torch.tensor([[1.,0.],[0.,1.]])\ni=torch.tensor([[1.,0.],[.5,.5],[0.,1.]])\ne=torch.tensor([[0,0,1],[2,3,4]])\nresult=evaluate(u,i,e,2)',
        'precision_u = hits / k',
        ['precision_u = hits / n_pos', 'precision_u = k / hits', 'precision_u = hits / len(topk_indices + pos_items)'],
        'Precision@K의 분모는 추천한 항목 수 K이고 Recall의 분모는 실제 관련 항목 수입니다.',
        setup='from collections import defaultdict')

    question('rag-reader-prompt', 'RAG', 'medium', 'RAG Reader: 검색 근거만 사용하는 Prompt', 'RAG 구성 이해 및 활용',
        RAG + '2일차/code/2. Task_1.ipynb', 29, 'Reader.prompt_generator',
        '검색된 chunk를 References로 묶고 길이를 제한한 뒤 질문과 함께 LLM 메시지로 구성하세요.',
        'MAX_CONTEXT_REFERENCES_LENGTH=4000\nself=NS(system_prompt="system")\nresult=prompt_generator(self,"question",["ref-a","ref-b-"+str(case)])',
        'references = references[:MAX_CONTEXT_REFERENCES_LENGTH]',
        ['references = references[MAX_CONTEXT_REFERENCES_LENGTH:]', 'references = ""', 'references = query[:MAX_CONTEXT_REFERENCES_LENGTH]'],
        'Reader는 검색 근거를 context 한도까지만 잘라 질문과 함께 전달합니다.')

    question('rag-crag-score', 'RAG', 'easy', 'RAG 평가: 정답·보류·환각 점수', 'RAG 평가',
        RAG + '2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb', 36, 'record_result',
        'perfect에는 1점, acceptable에는 0.5점, incorrect에는 -1점을 주어 CRAG score를 계산하세요.',
        'RESULTS={}\ndef save_results():pass\ncounts={"perfect":3+case,"acceptable":2,"missing":1,"incorrect":1}\nresult=record_result("x",counts,7+case,7,70,1.)',
        "crag_score = counts['perfect'] + 0.5 * counts['acceptable'] - counts['incorrect']",
        ["crag_score = counts['perfect'] + counts['acceptable']", "crag_score = counts['perfect'] - counts['missing']", "crag_score = counts['perfect'] + 0.5 * counts['acceptable'] + counts['incorrect']"],
        'CRAG score는 환각성 오답을 감점하고, 허용 가능한 답에는 절반 점수를 줍니다.')
