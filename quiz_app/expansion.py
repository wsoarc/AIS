"""Additional core implementations, extracted from the original practice notebooks."""

def register(add, extract, LLM, VIS, OD, DATA, RAG):
    from quiz_app.mcq import entry

    def question(id, field, level, title, point, path, index, symbol, prompt,
                 fixture, focus, alternatives, explanation, setup=''):
        # Both formats use the identical source implementation and CPU fixtures.
        add('core-'+id, field, level, title, point, path, index, symbol=symbol,
            prompt=prompt, fixture=fixture, wrong=[(focus, a) for a in alternatives],
            hint=explanation, explanation=explanation, setup=setup,
            mode='호출 흐름 실행 · 외부 서비스 대체 객체' if field=='RAG' else 'CPU 실행')
        entry(id, focus, prompt, alternatives)

    question('llm-multihead','LLM','hard','Multi-head Attention: 헤드 분리와 재결합','모델의 원리 이해',
        LLM+'Chapter_3_Excercise_Attention.ipynb',3,'MultiHeadAttention.forward',
        '입력 (B,T,D)을 여러 head로 나누어 causal attention을 수행하고 (B,T,D_out)으로 합치세요. num_heads × head_dim = d_out이며 dropout은 비활성화합니다.',
        '''d=4+case
x=torch.randn(2,3+case,d)
self=NS(W_key=nn.Linear(d,8),W_query=nn.Linear(d,8),W_value=nn.Linear(d,8),num_heads=2,head_dim=4,d_out=8,mask=torch.triu(torch.ones(8,8),diagonal=1),dropout=nn.Identity(),out_proj=nn.Linear(8,8))
result=forward(self,x)''',
        'context_vec = (attn_weights @ values).transpose(1, 2)',
        ['context_vec = (attn_weights @ values).transpose(2, 3)','context_vec = (attn_weights @ keys).transpose(1, 2)','context_vec = attn_weights @ values'],
        '가중합은 (B,H,T,D_head)입니다. 토큰 축을 head 축 앞으로 옮긴 뒤 head들을 합쳐야 토큰별 표현을 유지합니다.')

    question('llm-gpt','LLM','medium','GPT: 토큰·위치 임베딩에서 어휘 logits까지','모델의 원리 이해',
        LLM+'Chapter_4_Excercise_GPT.ipynb',3,'GPTModel.forward',
        '토큰 ID (B,T)를 입력받아 토큰·위치 임베딩, Transformer, 최종 정규화와 출력 head를 연결하세요. 출력은 (B,T,V) logits입니다.',
        '''self=NS(tok_emb=nn.Embedding(11,4),pos_emb=nn.Embedding(8,4),drop_emb=nn.Identity(),trf_blocks=nn.Sequential(nn.Linear(4,4),nn.GELU()),final_norm=nn.LayerNorm(4),out_head=nn.Linear(4,11))
result=forward(self,torch.randint(0,11,(2,3+case)))''',
        'x = tok_embeds + pos_embeds',
        ['x = tok_embeds','x = tok_embeds * pos_embeds','x = torch.cat((tok_embeds, pos_embeds.expand_as(tok_embeds)), dim=-1)'],
        '동일한 D차원의 토큰·위치 표현을 더해 토큰 의미와 순서를 함께 전달합니다. 출력 head는 각 위치를 어휘 차원으로 투영합니다.')

    question('llm-collate','LLM','hard','Instruction 학습: 패딩과 손실 마스킹','데이터 구성 원리 이해',
        LLM+'Chapter_7_Exercise_Follow_Instructions.ipynb',0,'custom_collate_fn',
        '길이가 다른 토큰 목록을 배치로 구성하세요. next-token 입력·정답을 만들고 첫 EOS는 학습하되 이후 패딩 정답은 ignore_index로 제외하세요. 최대 길이 제한도 반영합니다.',
        'result=custom_collate_fn([[1,2,3,4],[2],[4,3]],pad_token_id=9,allowed_max_length=None if case==0 else 2+case)',
        'targets[indices[1:]] = ignore_index',
        ['targets[indices] = ignore_index','targets[indices[1:]] = pad_token_id','inputs[indices[1:]] = ignore_index'],
        '첫 종료 토큰은 생성 종료를 배우는 정답입니다. 그 이후 길이 맞춤용 패딩만 정답 손실에서 제외합니다.')

    question('llm-classification-loss','LLM','medium','분류 미세조정: 시퀀스 표현과 클래스 정답','모델의 원리 이해',
        LLM+'Chapter_6_Excercise_Finetuning_Classification.ipynb',0,'calc_loss_batch',
        '모델은 (B,T,C) logits를 반환합니다. 실습처럼 마지막 토큰의 출력을 문장 분류에 사용하여 (B,) 클래스 정답과 cross entropy를 계산하세요.',
        'result=calc_loss_batch(torch.randint(0,9,(3,2+case)),torch.tensor([0,1,2]),nn.Embedding(9,3),"cpu")',
        'logits = model(input_batch)[:, -1, :]',
        ['logits = model(input_batch)[:, 0, :]','logits = model(input_batch).mean(dim=1)','logits = model(input_batch)'],
        '분류 정답은 문장당 하나입니다. 이 실습에서는 마지막 토큰의 C차원 출력을 정답 클래스와 연결합니다.')

    question('llm-dpo','LLM','hard','DPO: 선호·비선호 응답과 기준 모델','모델의 원리 이해',
        LLM+'Chapter_7_Exercise_Follow_Instructions_dpo.ipynb',0,'compute_dpo_loss',
        '선호/비선호 응답의 로그확률로 policy와 reference의 로그비를 계산하고 beta를 반영한 DPO 손실을 반환하세요. 함께 반환하는 두 reward는 gradient를 분리합니다.',
        '''a=torch.randn(3+case,requires_grad=True);b=torch.randn(3+case,requires_grad=True)
out=compute_dpo_loss(a,b,torch.randn(3+case),torch.randn(3+case),.2+case)
out[0].backward()
result=(out,a.grad,b.grad,out[1].requires_grad,out[2].requires_grad)''',
        'logits = model_logratios - reference_logratios',
        ['logits = model_logratios + reference_logratios','logits = reference_logratios - model_logratios','logits = model_logratios'],
        '현재 모델의 선호 차이를 기준 모델의 선호 차이와 비교합니다. 기준 모델 대비 선호 응답의 상대 확률을 높이는 목적입니다.')

    question('vis-up','Vision','hard','U-Net 디코더: 해상도 복원과 skip 특징 결합','모델의 원리 이해',
        VIS+'04_Unet.ipynb',6,'Up.forward',
        '제공된 up과 conv로 디코더 블록을 완성하세요. x1을 업샘플하고 x2와 공간 크기를 맞춘 후 [skip, decoder] 순서로 채널 결합합니다. 홀수 해상도도 처리해야 합니다.',
        '''self=NS(up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),conv=nn.Conv2d(5,2,3,padding=1))
result=forward(self,torch.randn(2,2,3,4),torch.randn(2,3,6+case,8+case))''',
        'x = torch.cat([x2, x1], dim=1)',
        ['x = torch.cat([x2, x1], dim=2)','x = torch.cat([x1, x2], dim=1)','x = x2 + x1'],
        '공간 크기를 정렬한 다음 채널 축으로 결합합니다. 후속 conv는 skip과 업샘플 특징을 함께 받아 위치 정보를 복원합니다.')

    question('vis-transformer','Vision','medium','ViT Encoder: Attention·FFN의 연속 잔차 갱신','모델의 원리 이해',
        VIS+'02_ViT_CIFAR10.ipynb',8,'Transformer.forward',
        '각 layers 항목은 pre-norm attention과 FFN입니다. attention 잔차 갱신 결과를 다음 FFN의 입력과 잔차로 사용하며 모든 층을 순서대로 통과시키세요.',
        'self=NS(layers=[(nn.Linear(4,4),nn.Sequential(nn.Linear(4,4),nn.ReLU())) for _ in range(2+case)])\nresult=forward(self,torch.randn(2,3,4))',
        'x = ff(x) + x',
        ['x = ff(x)','x = ff(x) - x','x = attn(x) + x'],
        'FFN은 attention으로 갱신한 토큰 표현에 작용하며 잔차로 기존 표현과 gradient 경로를 보존합니다.')

    question('vis-validation','Vision','medium','분류 검증: 평가 모드와 전체 샘플 평균','평가 Metric 이해',
        VIS+'01_ResNet18_CIFAR10.ipynb',10,'evaluate',
        '평가 모드에서 분류 loss와 정확도를 집계하세요. 마지막 배치 크기가 다를 수 있으므로 샘플 수로 가중한 전체 평균을 반환합니다. device와 accuracy_top1은 제공됩니다.',
        '''model=nn.Sequential(nn.Linear(4,3),nn.Dropout(.7))
loader=[(torch.randn(3,4),torch.tensor([0,1,2])),(torch.randn(1+case,4),torch.arange(1+case)%3)]
result=(evaluate(model,loader,nn.CrossEntropyLoss()),model.training)''',
        'total_loss += loss.item() * bs',
        ['total_loss += loss.item()','total_loss += loss.item() / bs','total_loss = loss.item() * bs'],
        '배치 평균 loss에 배치 샘플 수를 곱해 합산한 뒤 전체 샘플 수로 나눕니다. 작은 마지막 배치를 동일 비중으로 평균하면 왜곡됩니다.',
        setup='device=torch.device("cpu")\n'+extract(VIS+'01_ResNet18_CIFAR10.ipynb',10,'accuracy_top1'))

    question('od-linear','On-device','hard','양자화 Linear: 정수 누산과 출력 재양자화','Quantization',
        OD+'1. Basic Quantization Lab_answer.ipynb',55,'quantized_linear',
        'int8 입력·가중치와 int32 shifted bias로 Linear를 계산하세요. 입력 zero point는 bias에 이미 반영되어 있습니다. 채널별 weight scale과 입출력 scale, 출력 zero point로 재양자화합니다. 채점은 CPU입니다.',
        '''result=quantized_linear(torch.tensor([[12,-8,5],[-4,9,2]],dtype=torch.int8),torch.tensor([[3,-5,2],[-2,7,4]],dtype=torch.int8),torch.tensor([3,-7],dtype=torch.int32),8,8,0,3+case,.3,torch.tensor([.2,.7]).view(2,1),.05+case*.01)''',
        'output = output * (input_scale * weight_scale / output_scale)',
        ['output = output * (input_scale * weight_scale * output_scale)','output = output * (input_scale / weight_scale / output_scale)','output = output * (weight_scale / output_scale)'],
        '정수 누산값의 실수 단위는 input_scale × weight_scale입니다. 이를 output_scale로 나누어 출력 정수 단위로 바꿉니다.',
        setup=extract(OD+'1. Basic Quantization Lab_answer.ipynb',22,'get_quantized_range'))

    question('od-smooth','On-device','hard','SmoothQuant: 활성값·가중치 범위의 균형','Quantization',
        OD+'2. Advanced Quantization Lab_answer.ipynb',55,'smooth_ln_fcs',
        'act_scales와 후속 Linear들의 입력 채널별 최대 가중치로 smoothing scale을 구성하세요. LN 파라미터를 나누고 Linear 입력 채널을 곱하여 함수는 보존하면서 범위를 이동합니다. no_grad에서 호출됩니다.',
        '''ln=nn.LayerNorm(4);ln.bias.data.copy_(torch.tensor([.1,.2,-.1,.4]))
fcs=[nn.Linear(4,3),nn.Linear(4,2)];x=torch.randn(2,4)
with torch.no_grad(): smooth_ln_fcs(ln,fcs,torch.tensor([.2,2.,5.,1.]),alpha=.2+case*.3)
result=(ln.weight,ln.bias,[fc.weight for fc in fcs],[fc(ln(x)) for fc in fcs])''',
        'scales = act_scales.pow(alpha) / weight_scales.pow(1 - alpha)',
        ['scales = act_scales.pow(alpha) * weight_scales.pow(1 - alpha)','scales = weight_scales.pow(1 - alpha) / act_scales.pow(alpha)','scales = act_scales / weight_scales'],
        'alpha로 activation과 weight 범위의 이동량을 조절합니다. LN 출력의 축소와 다음 Linear 가중치의 확대를 짝지어 원래 연산을 보존합니다.')

    question('od-activation','On-device','medium','활성값 양자화: 토큰별 absmax scale','Quantization',
        OD+'2. Advanced Quantization Lab_answer.ipynb',43,'quantize_activation_per_token_absmax',
        '활성값 t=(B,T,D)에 대해 토큰별 마지막 특성 축 absmax로 scale을 구하고 양자화·복원하세요. n_bits는 주어지며 영벡터에서도 0으로 나누지 않아야 합니다.',
        '''t=torch.randn(2,3+case,4)*torch.tensor([.1,1.,3.,8.]);t[0,0]=0
result=quantize_activation_per_token_absmax(t,n_bits=3+case)''',
        'scales = t.abs().max(dim=-1, keepdim=True)[0]',
        ['scales = t.abs().max()','scales = t.abs().max(dim=1, keepdim=True)[0]','scales = t.abs().mean(dim=-1, keepdim=True)'],
        '각 토큰은 D개 특성의 최대 절댓값으로 별도 scale을 갖습니다. keepdim은 원래 활성값에 대한 broadcasting을 유지합니다.')

    question('data-conv','Data','medium','Conv1D 시계열 모델: 시간·채널 축 연결','모델의 원리 이해',
        DATA+'3_4-ts-practice/ts_solution.ipynb',29,'Conv1DModel.forward',
        '입력은 (B,T,C)이고 제공된 conv1d는 C→H, kernel_size=2입니다. Conv1d 입력 축으로 바꿔 시간 특징을 추출하고 fc(H→1)로 시점별 예측을 반환하세요.',
        'self=NS(conv1d=nn.Conv1d(2,4,2),fc=nn.Linear(4,1))\nresult=forward(self,torch.randn(3,5+case,2))',
        'return self.fc(x)',
        ['return self.fc(x[:, -1])','return x','return self.fc(x).mean(dim=1)'],
        'Conv1d 앞뒤에서 시간·채널 축을 교환합니다. fc는 마지막 H축을 1로 투영하여 (B,T-1,1)을 유지합니다.')

    question('data-multistep','Data','medium','다중 시점 예측: encoder 입력과 미래 정답','데이터 구성 원리 이해',
        DATA+'3_4-ts-practice/ts_solution.ipynb',41,'create_enc_dec_sequences',
        '과거 sequence_length개 구간과 바로 다음 target_len개 미래 구간을 배치 텐서로 구성하세요. 원본 실습의 샘플 개수 N-sequence_length-target_len을 유지합니다. 데이터는 (N,C)입니다.',
        'sequence_length=3+case;target_len=2+case\nresult=create_enc_dec_sequences(np.arange(40,dtype=float).reshape(20,2))',
        'labels.append(data[i + sequence_length:i + sequence_length + target_len])',
        ['labels.append(data[i:i + target_len])','labels.append(data[i + sequence_length - 1:i + sequence_length + target_len - 1])','labels.append(data[i + sequence_length])'],
        'encoder 입력과 decoder 정답은 시간상 겹치지 않습니다. 정답은 단일 값이 아닌 미래 target_len개 시점의 시퀀스입니다.')

    question('data-encoder-decoder','Data','hard','Encoder–Decoder: 상태 전달과 자기회귀 예측','모델의 원리 이해',
        DATA+'3_4-ts-practice/ts_solution.ipynb',43,'RNNRNN.forward',
        'encoder의 최종 hidden state와 입력의 마지막 관측값으로 decoder를 시작하세요. 이후 이전 예측을 다음 입력으로 넣어 target_len개 시점의 예측 (B,target_len,C)을 반환합니다.',
        'self=NS(encoder=EncoderRNN(2,4,2),decoder=DecoderRNN(2,4,2))\nresult=forward(self,torch.randn(3,5,2),2+case)',
        'input = out',
        ['input = source[:, -1, :].unsqueeze(1)','input = torch.zeros_like(out)','input = out.mean(dim=0, keepdim=True)'],
        '다음 시점 입력은 방금 예측한 값입니다. hidden state도 매번 갱신하여 과거 정보와 연속된 예측을 연결합니다.',
        setup=extract(DATA+'3_4-ts-practice/ts_solution.ipynb',43,'EncoderRNN')+'\n'+extract(DATA+'3_4-ts-practice/ts_solution.ipynb',43,'DecoderRNN'))

    question('rag-synthesis','RAG','easy','LlamaIndex: 검색 노드와 응답 합성기 연결','관련 Library 활용',
        RAG+'1일차/code/1. Llama_index.ipynb',74,'StandardQueryEngine.custom_query',
        'Retriever가 반환한 Node 목록을 원래 질의와 함께 response_synthesizer에 전달하는 QueryEngine을 구현하세요. 인터페이스 대체 객체로 호출 흐름을 검증합니다.',
        'self=NS(retriever=NS(retrieve=lambda q:[q+"-node-"+str(i) for i in range(case+1)]),response_synthesizer=NS(synthesize=lambda q,n:{"query":q,"nodes":n}))\nresult=custom_query(self,"question")',
        'response_obj = self.response_synthesizer.synthesize(query_str, nodes)',
        ['response_obj = self.response_synthesizer.synthesize(query_str, [])','response_obj = self.response_synthesizer.synthesize(nodes, query_str)','response_obj = self.response_synthesizer.synthesize(query_str, nodes[::-1])'],
        '검색 노드는 응답 합성기의 근거입니다. 질의와 근거를 함께 전달해야 검색 결과가 최종 답변에 반영됩니다.')

    question('rag-context','RAG','medium','Custom QueryEngine: 검색 근거에서 생성 프롬프트까지','RAG 구성 이해 및 활용',
        RAG+'1일차/code/1. Llama_index.ipynb',79,'OurCustomQueryEngine.custom_query',
        '검색된 노드의 내용을 문맥으로 모으고 질문과 함께 qa_prompt에 넣어 LLM에 전달하세요. 검색→문맥 구성→생성의 전체 흐름을 구현합니다. 외부 모델 호출은 대체 객체로 검증합니다.',
        '''self=NS(retriever=NS(retrieve=lambda q:[NS(node=NS(get_content=lambda i=i:'evidence-'+str(i))) for i in range(1+case)]),qa_prompt=NS(format=lambda **kw:kw),llm=NS(complete=lambda p:repr(p)))
result=custom_query(self,'question')''',
        'response = self.llm.complete(self.qa_prompt.format(context_str=context_str, query_str=query_str))',
        ['response = self.llm.complete(self.qa_prompt.format(context_str="", query_str=query_str))','response = self.llm.complete(self.qa_prompt.format(context_str=query_str, query_str=context_str))','response = self.llm.complete(query_str)'],
        'LLM 입력에는 질문과 검색 근거가 함께 있어야 합니다. 노드 객체 자체가 아니라 각 노드의 실제 내용을 문맥으로 구성합니다.')

    question('rag-verification','RAG','medium','RAG 검증: 근거가 뒷받침하는 답변만 반환','RAG 구성 이해 및 활용',
        RAG+'2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb',40,'infer_L2',
        'RAG 응답을 질문·검색 근거·후보 답변으로 검증하고 supported일 때만 답변을 반환하세요. 그렇지 않으면 실습처럼 I don\'t know를 반환합니다. 검증기는 제공됩니다.',
        '''calls=[]
rag=NS(inference=lambda q,s,k:{'answer':'candidate','retrieved_results':s[:k]})
def verify_answer(q,refs,a):
    calls.append((q,refs,a))
    return case==0,''
result=(infer_L2({'query':'question','search_results':['ref-'+str(case)]}),calls)''',
        '''return out['answer'] if supported else "I don't know"''',
        ['return out["answer"]','return "I don\'t know"','return out["answer"] if not supported else "I don\'t know"'],
        '그럴듯한 답변 여부와 근거의 지원 여부는 다릅니다. 이 실습은 근거 검증 실패 시 답변을 보류하는 흐름을 구현합니다.')
