"""Short, curated blanks for MCQ; full implementations remain code questions."""
import textwrap

# Each blank tests a core decision in the surrounding notebook implementation.
# Alternatives replace ONLY this block, not the whole function.
ITEMS = {}
def entry(id, focus, prompt, alternatives):
    ITEMS['core-'+id] = (focus, prompt, alternatives)

entry('llm-layernorm',
    'norm_x = (x - mean) / torch.sqrt(var + self.eps)',
    '각 토큰의 특성을 평균 0, 분산 1에 가깝게 정규화하려 합니다. 평균·분산이 계산된 뒤 빈칸에 들어갈 코드를 고르세요.',
    ['norm_x = (x + mean) / torch.sqrt(var + self.eps)',
     'norm_x = (x - mean) / (var + self.eps)',
     'norm_x = x / torch.sqrt(var + self.eps)'])
entry('llm-training','loss.backward()\noptimizer.step()',
    '손실을 계산했습니다. 이 손실이 모델 파라미터 갱신으로 이어지도록 빈칸을 채우세요.',
    ['optimizer.step()\nloss.backward()', 'loss.backward()\noptimizer.zero_grad()', 'optimizer.zero_grad()\noptimizer.step()'])
entry('llm-dataset','target_chunk = token_ids[i + 1:i + max_length + 1]',
    '입력 토큰마다 다음 토큰을 예측하도록 정답 시퀀스를 구성하려 합니다. 입력과 같은 길이의 타깃을 만드는 코드를 고르세요.',
    ['target_chunk = token_ids[i:i + max_length]', 'target_chunk = token_ids[i + 1:i + max_length]', 'target_chunk = token_ids[i + max_length:i + 2 * max_length]'])
entry('llm-attention','attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)',
    '미래 위치가 True인 causal mask가 주어졌습니다. softmax 이후 미래 토큰의 가중치가 0이 되도록 빈칸을 채우세요.',
    ['attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], 0.0)',
     'attn_scores.masked_fill_(~self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)',
     'attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], torch.inf)'])
entry('llm-transformer','x = self.norm1(x)\nx = self.att(x)',
    '노트북의 pre-norm Transformer입니다. 잔차 입력을 보관한 다음 Attention 경로를 처리하는 순서를 고르세요.',
    ['x = self.att(x)\nx = self.norm1(x)', 'x = self.norm1(x)\nx = self.ff(x)', 'x = self.att(x)\nx = self.att(x)'])
entry('vis-evaluation','per_class_acc = cm.diagonal().float() / torch.clamp(cm.sum(dim=1).float(), min=1.0)',
    '혼동행렬의 행은 정답, 열은 예측입니다. 정답 샘플이 없는 클래스는 0으로 처리하면서 클래스별 정확도를 계산하는 코드를 고르세요.',
    ['per_class_acc = cm.diagonal().float() / torch.clamp(cm.sum(dim=0).float(), min=1.0)',
     'per_class_acc = cm.diagonal().float() / cm.sum().clamp(min=1)',
     'per_class_acc = cm.sum(dim=1).float() / torch.clamp(cm.diagonal().float(), min=1.0)'])
entry('vis-preprocessing','test_tfms = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])',
    '학습 전처리가 제공되어 있습니다. 학습과 같은 입력 척도를 유지하되 무작위 증강을 제거한 평가 전처리를 고르세요.',
    ['test_tfms = train_tfms', 'test_tfms = T.Compose([T.ToTensor()])',
     'test_tfms = T.Compose([T.Normalize(CIFAR10_MEAN, CIFAR10_STD), T.ToTensor()])'])
entry('vis-training','loss = criterion(logits, masks)',
    'criterion은 BCEWithLogitsLoss이고, masks는 실제 0/1 분할 정답입니다. 학습 목표에 맞게 예측과 정답을 연결하세요.',
    ['loss = criterion(masks, logits)', 'loss = criterion(logits.sigmoid(), masks)', 'loss = criterion(logits, logits.detach())'])
entry('vis-vit',"x = x[:, 0] if self.pool == 'cls' else x.mean(dim=1)",
    'Transformer 출력 x는 (B,N,D)이고 첫 토큰은 CLS입니다. 설정에 따라 CLS 표현 또는 토큰 평균을 분류기에 넘기는 코드를 고르세요.',
    ["x = x[:, -1] if self.pool == 'cls' else x.mean(dim=1)",
     "x = x[:, 0] if self.pool == 'cls' else x.mean(dim=0)",
     "x = x.mean(dim=1) if self.pool == 'cls' else x[:, 0]"])
entry('vis-unet','x = self.up1(x5, x4)',
    '가장 깊은 특징 x5부터 첫 디코더 단계를 시작합니다. 해상도가 대응하는 encoder 특징을 skip으로 연결하는 코드를 고르세요.',
    ['x = self.up1(x5, x3)', 'x = self.up1(x4, x5)', 'x = self.up1(x5, x5)'])
entry('od-quantization','w = (w - zeros) * scales',
    'w는 정수 코드로 양자화·clipping된 값입니다. 같은 zero point와 scale을 사용하여 실수 가중치로 복원하는 코드를 고르세요.',
    ['w = (w + zeros) * scales', 'w = (w - zeros) / scales', 'w = w * scales + zeros'])
entry('od-perchannel','_subtensor = tensor.select(dim_output_channels, oc)',
    '각 출력 채널의 가중치 범위로 별도의 scale을 계산하려 합니다. 이번 채널의 가중치를 선택하는 코드를 고르세요.',
    ['_subtensor = tensor', '_subtensor = tensor.select(1, oc)', '_subtensor = tensor.mean(dim=dim_output_channels)'])
entry('od-pruning','scores = attn_kv.sum(dim=-2)',
    'attn_kv는 (B,H,Q,K)입니다. 각 key 토큰이 모든 query에서 받은 attention을 누적하여 중요도를 계산하세요.',
    ['scores = attn_kv.sum(dim=-1)', 'scores = attn_kv.sum(dim=1)', 'scores = attn_kv[..., -1, :]'])
entry('od-speculative','break',
    'Draft와 Target을 앞에서부터 비교하고 있습니다. 첫 불일치를 만났을 때 연속으로 일치하는 prefix만 수락하려면 빈칸에서 무엇을 해야 할까요?',
    ['continue', 'n = 0', 'n += 1'])
entry('od-distillation','loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss',
    'Teacher 분포와의 soft target 손실, 실제 정답의 label 손실이 계산되어 있습니다. 두 학습 목표를 주어진 비중대로 Student 학습에 반영하세요.',
    ['loss = soft_target_loss_weight * soft_targets_loss', 'loss = ce_loss_weight * label_loss',
     'loss = soft_target_loss_weight * soft_targets_loss - ce_loss_weight * label_loss'])
entry('data-window','labels.append(data_seq[i + sequence_length, 0])',
    'features에는 과거 sequence_length개 값이 들어갑니다. 그 구간 직후의 미래 값이 정답이 되도록 빈칸을 채우세요.',
    ['labels.append(data_seq[i + sequence_length - 1, 0])', 'labels.append(data_seq[i, 0])', 'labels.append(data_seq[i:i + sequence_length, 0])'])
entry('data-scaling','test_scaled = scaler.transform(test_data.values)',
    'scaler는 학습 데이터에 이미 fit했습니다. 평가 데이터 누수 없이 같은 척도를 적용하는 코드를 고르세요.',
    ['test_scaled = scaler.fit_transform(test_data.values)', 'test_scaled = test_data.values', 'test_scaled = scaler.transform(train_data.values)'])
entry('data-evaluation','rmse = np.sqrt(mean_squared_error(y_test, test_predictions))',
    '원래 정답 값과 같은 단위로 예측 오차를 해석하려고 RMSE를 계산합니다. 빈칸에 들어갈 코드를 고르세요.',
    ['rmse = mean_squared_error(y_test, test_predictions)', 'rmse = mean_absolute_percentage_error(y_test, test_predictions)', 'rmse = np.mean(y_test - test_predictions)'])
entry('data-lstm','return self.linear(out)',
    'LSTM은 (B,T,H) 특징을 반환하고 linear는 H→1입니다. 각 시점의 예측을 유지하여 (B,T,1)을 반환하는 코드를 고르세요.',
    ['return self.linear(out[:, -1])', 'return out', 'return self.linear(out).mean(dim=1)'])
entry('data-bpr','loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))',
    'BPR은 선호 아이템의 점수가 비선호 아이템보다 높아지도록 학습합니다. 최소화할 순위 손실을 고르세요.',
    ['loss = -torch.mean(F.logsigmoid(neg_scores - pos_scores))', 'loss = torch.mean(F.logsigmoid(pos_scores - neg_scores))', 'loss = torch.mean((pos_scores - neg_scores) ** 2)'])
entry('data-training','pred = model(batch_x)[:, -1, 0]',
    '모델 출력은 (B,T,1), batch_y는 각 윈도우 다음 시점의 정답 (B,)입니다. 정답과 대응시킬 예측을 고르세요.',
    ['pred = model(batch_x)[:, 0, 0]', 'pred = model(batch_x).mean(dim=1)[:, 0]', 'pred = model(batch_x)[:, :, 0]'])
entry('rag-pipeline','answer = self.reader.generate_response(query, retrieved_results)',
    'Retriever가 관련 근거를 선택했습니다. 원래 질문과 선택된 근거가 Reader의 답변 생성으로 이어지도록 빈칸을 채우세요.',
    ['answer = self.reader.generate_response(query, search_results)', 'answer = self.reader.generate_response(query, [])', 'answer = self.reader.generate_response(retrieved_results, query)'])
entry('rag-mcp','retrieved_results = await self.retrieve(query, query_time, search_results, topk)',
    'retrieve는 async 함수입니다. 질문·질문 시점·검색 자료를 전달하고 완료된 근거를 다음 생성 단계에 넘기는 코드를 고르세요.',
    ['retrieved_results = self.retrieve(query, query_time, search_results, topk)',
     'retrieved_results = await self.retrieve(query, search_results, query_time, topk)',
     'retrieved_results = await self.retrieve(query, query_time, [], topk)'])
entry('rag-llamaindex','base_index = VectorStoreIndex.from_documents(documents=documents, transformations=[self.parser])',
    '문서를 제공된 parser로 분할·인덱싱한 뒤 검색하려 합니다. 원본 문서와 전처리가 검색 인덱스에 연결되도록 빈칸을 채우세요.',
    ['base_index = VectorStoreIndex.from_documents(documents=documents, transformations=[])',
     'base_index = VectorStoreIndex.from_documents(documents=[], transformations=[self.parser])',
     'base_index = VectorStoreIndex.from_documents(documents=[query], transformations=[self.parser])'])
entry('rag-hybrid','combined_results = [kg_results]',
    '금융 질의이면 KG 결과를 근거로 사용한다는 실습 정책입니다. 해당 분기에서 Reader에 전달할 근거를 고르세요.',
    ['combined_results = retrieved_results', 'combined_results = []', 'combined_results = [query]'])
entry('rag-retriever','top_k_indices = (-cosine_scores).argsort()[:topk]',
    '각 청크와 질문의 cosine similarity가 계산되었습니다. 관련성이 높은 순으로 topk개 청크의 인덱스를 선택하세요.',
    ['top_k_indices = cosine_scores.argsort()[:topk]', 'top_k_indices = np.arange(topk)', 'top_k_indices = (-cosine_scores).argsort()[-topk:]'])


def attach(bank, assemble, normalized, marker):
    if not {q['id'] for q in bank}.issubset(ITEMS):
        raise ValueError('Every question must have a curated MCQ blank')
    for q in bank:
        focus, prompt, alternatives = ITEMS[q['id']]
        answer = normalized(focus)
        options = [answer] + [normalized(a) for a in alternatives]
        if len(set(options)) != 4 or any(len(o.splitlines()) > 3 for o in options):
            raise ValueError((q['id'], 'MCQ options must be short and distinct'))
        reference = assemble(q, q['answer']).rstrip() + '\n'
        matches=[]
        for width in range(0, 21, 4):
            block=textwrap.indent(answer,' '*width)
            # Match full lines so an inner block is not confused with a parent.
            start=reference.find('\n'+block+'\n')
            if start >= 0:
                matches.append((width,block))
        if not matches: raise ValueError((q['id'],'MCQ blank missing',answer))
        width,block=max(matches)
        if reference.count('\n'+block+'\n') != 1: raise ValueError((q['id'],'Ambiguous MCQ blank'))
        template=reference.replace('\n'+block+'\n','\n'+' '*width+marker+'\n',1)
        q['mcq']={**q,'template':template,'answer':answer,'indent':width,
                  'options':options,'prompt':prompt+'\n주변 코드는 제공되어 있습니다. 빈칸에 들어갈 코드만 선택하세요.'}
