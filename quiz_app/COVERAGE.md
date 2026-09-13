# Summary 기준 문제은행

총 86문항. 객관식은 주변 실습 코드의 핵심 연산·연결을 선택하고, 주관식은 같은 구현의 함수·블록을 작성합니다. 객관식/주관식 변환을 별도 문항으로 세지 않습니다.

## 범위와 묶음 기준

다음 다섯 파일의 **코드가 있는 주제**를 실제 노트북 구현과 연결했습니다. 모든 줄을 독립 문항으로 만들지 않고 데이터 구성, 모델 구성, 학습, 검색·생성 흐름으로 묶었습니다. 아래 표에서 “포함”은 구현 문제 또는 제공되는 주변 코드에 포함된다는 뜻이며 모든 세부 연산이 각각 객관식 빈칸이라는 뜻은 아닙니다.

- LLM: `1. LLM/example_ipython/llm_summary.txt`
- Vision: `2. Vision/example_ipython/vision_summary.txt`
- On-device: `3. On-device/example_ipython/on-device_summary.txt`
- Data: `4. Data/data_summary.txt`
- RAG: `5. RAG/실습자료_Colab/rag_summary.txt`

## Summary 항목 대응

문항 ID는 아래 표에서 공통 접두사 `core-`를 생략했습니다.

| 분야 | Summary 주제 | 대응 문항 |
|---|---|---|
| LLM | tokenizer 인코딩·디코딩 | llm-token-roundtrip |
| LLM | 입력/정답 sliding window | llm-dataset |
| LLM | Embedding 생성·적용 | llm-embedding-build, llm-gpt |
| LLM | Causal / Multi-head Attention | llm-attention, llm-multihead |
| LLM | LayerNorm·Transformer·GPT 구성 | llm-layernorm, llm-transformer, llm-embedding-build, llm-gpt |
| LLM | 토큰 생성·문자열 복원 | llm-generation, llm-token-roundtrip |
| LLM | 모든 토큰의 CE loss·학습 갱신 | llm-token-loss, llm-training |
| LLM | LoRA 초기화·forward·원본 동결 | llm-lora-init, llm-lora, llm-linear-lora, llm-lora-freeze |
| Vision | transforms·CIFAR10 dataset | vis-preprocessing, vis-dataset |
| Vision | ResNet CIFAR stem·pool·head 수정 | vis-resnet-cifar |
| Vision | ViT Attention·Transformer | vis-attention, vis-transformer |
| Vision | 패치 임베딩·CLS·위치·분류 head | vis-patch-build, vis-vit |
| Vision | DoubleConv·Down·Up·UNet 구성 | vis-doubleconv, vis-down, vis-up, vis-unet-build, vis-unet |
| On-device | Linear 양자화·scale/zero point·weight scale | od-quantization, od-scale-zero, od-perchannel |
| On-device | FC/Conv 출력의 scale·zero point 재양자화 | od-linear |
| On-device | K-means codebook·QAT centroid 갱신 | od-kmeans, od-qat-codebook |
| On-device | AWQ 중요 채널·활성값 기반 scale | od-awq, od-awq-scales |
| On-device | Weight/Activation 범위 이동 | od-smooth |
| On-device | Rotation의 입력·출력 축 | od-rotation |
| On-device | Soft-target·cosine·hint KD | od-distillation, od-kd-cosine, od-kd-hint |
| On-device | KV memory·StreamingLLM | od-cache-memory, od-streaming |
| On-device | H2O·H2O-Norm·SnapKV | od-pruning, od-h2o-norm, od-snapkv |
| On-device | Draft 생성·Target 검증·prefix 수락 | od-draft, od-verify, od-speculative |
| On-device | DFlash 병렬 block 생성 | od-dflash |
| Data | 시간 분할·sliding window | data-time-split, data-window |
| Data | LSTM·CNN·RNN | data-lstm, data-conv, data-rnn |
| Data | Encoder/Decoder 상태·자기회귀 | data-encoder-decoder, data-rnn |
| Data | RMSE·MAPE | data-evaluation, data-mape |
| Data | LabelEncoder·층화 분할 | data-recsys-split |
| Data | NCF embedding·MLP·평점 학습 | data-ncf, data-ncf-train |
| Data | Graph edge 생성·분할 | data-graph-edges, data-graph-split |
| Data | NGCF 메시지·정규화·여러 깊이의 표현 결합 | data-ngcf-layer, data-ngcf-representation |
| Data | BPR loss·정규화·학습 연결 | data-bpr, data-bpr-train |
| Data | 순위 점수·Recall/Precision/NDCG | data-ranking-metrics, data-ndcg |
| RAG | 문서 로딩·chunking·index | rag-load-chunk, rag-llamaindex |
| RAG | Retriever·Synthesizer·QueryEngine | rag-synthesis, rag-context |
| RAG | 기본 검색→문맥→생성 | rag-pipeline, rag-context |
| RAG | Embedding 호출·cosine·top-k | rag-embedding-call, rag-retriever |
| RAG | HTML 본문 파싱 | rag-html |
| RAG | Reader prompt·생성 호출 | rag-reader-prompt, rag-reader-generate |
| RAG | KG 근거와 웹 근거 선택 | rag-hybrid |
| RAG | MCP Client·ToolSpec·tools/resources | rag-mcp-discovery |
| RAG | FunctionAgent·Context·비동기 실행 | rag-mcp-agent, rag-mcp-query |
| RAG | MCP RAG 검색·질의 시각·생성 연결 | rag-mcp, rag-mcp-time |
| RAG | exact/acceptable/missing/incorrect·CRAG score | rag-grade, rag-crag-score |

## 제외와 교정

- summary에서 패스한 instruction following/DPO, EAGLE은 제외합니다. DETR은 제목만 있고 코드가 없어 문제를 만들지 않습니다.
- 현재 summary에 없는 MinMaxScaler, 별도 시계열 학습/다중 시점 데이터 구성, 토큰별 absmax 양자화, RAG self-verification 문항을 제거했습니다. 요약에 있는 추천 학습과 RNN encoder–decoder는 포함합니다.
- summary의 철자·슬라이싱 오타는 문제로 옮기지 않고 실습 구현을 사용합니다. 예: tokenizer encode/decode, SnapKV의 마지막 window `-w:`.
- NGCF 정답 노트북의 사용자 방향 `index_add` 반환값 누락은 summary와 같은 `index_add_`로 교정했습니다. 원본 셀 SHA-256과 교정 내역은 문항 source에 기록합니다.
- 설정 상수·모델 이름·API 인자명은 제공합니다. 단순 이름 암기가 아니라 입력·출력·상태·학습 연결을 평가합니다.
- 외부 API, 모델 다운로드, CUDA 실행은 하지 않습니다. 작은 CPU tensor와 제공 객체로 동작을 검사합니다. Cosine KD의 원본 `.cuda()` 호출만 제공 wrapper로 CPU에 대응시킵니다.
- 기존 쉬움/보통/어려움 및 배점 5/10/15를 유지합니다. 시작은 객관식 20문항, 모의시험은 180분·분야별 4문항입니다.

## 문항별 실습 출처

셀 번호는 markdown 셀을 포함해 1부터 시작합니다.

| 분야 | ID | 문제 | 난이도 | 출처 |
|---|---|---|---|---|
| LLM | core-llm-layernorm | LayerNorm: 토큰별 정규화와 학습 파라미터 | easy | `1. LLM/example_ipython/Chapter_4_Excercise_GPT.ipynb` · 셀 3 · `LayerNorm.forward` |
| LLM | core-llm-linear-lora | LoRA Linear: 원본 출력과 Adapter 결합 | easy | `1. LLM/example_ipython/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb` · 셀 3 · `LinearWithLoRA.forward` |
| LLM | core-llm-token-roundtrip | Tokenizer: 문자열과 토큰 ID의 왕복 | easy | `1. LLM/example_ipython/Chapter_2_Exercise_Dataset.ipynb` · 셀 1 · `연속 코드 발췌` |
| LLM | core-llm-training | 언어 모델 학습: 손실에서 가중치 갱신까지 | medium | `1. LLM/example_ipython/Chapter_5_Excercise_Pretraining.ipynb` · 셀 3 · `연속 코드 발췌` |
| LLM | core-llm-dataset | 언어 모델 데이터셋: 슬라이딩 윈도우와 next-token 학습 | medium | `1. LLM/example_ipython/Chapter_2_Exercise_Dataset.ipynb` · 셀 3 · `GPTDatasetV1` |
| LLM | core-llm-gpt | GPT: 토큰·위치 임베딩에서 어휘 logits까지 | medium | `1. LLM/example_ipython/Chapter_4_Excercise_GPT.ipynb` · 셀 4 · `GPTModel.forward` |
| LLM | core-llm-generation | Greedy 생성: 마지막 토큰에서 다음 토큰 선택 | medium | `1. LLM/example_ipython/Chapter_4_Excercise_GPT.ipynb` · 셀 5 · `generate_text_simple` |
| LLM | core-llm-lora | LoRA: 저랭크 행렬과 스케일 적용 | medium | `1. LLM/example_ipython/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb` · 셀 3 · `LoRALayer.forward` |
| LLM | core-llm-embedding-build | GPT 구성: 토큰·위치 Embedding과 출력 Head | medium | `1. LLM/example_ipython/Chapter_4_Excercise_GPT.ipynb` · 셀 4 · `GPTModel` |
| LLM | core-llm-token-loss | 언어 모델: 모든 토큰의 정답과 손실 연결 | medium | `1. LLM/example_ipython/Chapter_5_Excercise_Pretraining.ipynb` · 셀 2 · `calc_loss_batch` |
| LLM | core-llm-lora-init | LoRA 초기화: 저랭크 경로와 초기 출력 보존 | medium | `1. LLM/example_ipython/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb` · 셀 3 · `LoRALayer` |
| LLM | core-llm-lora-freeze | LoRA 적용: 원본 동결과 Adapter 학습 | medium | `1. LLM/example_ipython/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb` · 셀 4 · `연속 코드 발췌` |
| LLM | core-llm-attention | Causal Attention: 토큰 간 정보 흐름 | hard | `1. LLM/example_ipython/Chapter_3_Excercise_Attention.ipynb` · 셀 1 · `CausalAttention.forward` |
| LLM | core-llm-transformer | Transformer block: 정규화·Attention·FFN·잔차 연결 | hard | `1. LLM/example_ipython/Chapter_4_Excercise_GPT.ipynb` · 셀 4 · `TransformerBlock.forward` |
| LLM | core-llm-multihead | Multi-head Attention: 헤드 분리와 재결합 | hard | `1. LLM/example_ipython/Chapter_3_Excercise_Attention.ipynb` · 셀 4 · `MultiHeadAttention.forward` |
| Vision | core-vis-preprocessing | 이미지 데이터: 학습 증강과 평가 전처리 | easy | `2. Vision/example_ipython/01_ResNet18_CIFAR10.ipynb` · 셀 6 · `연속 코드 발췌` |
| Vision | core-vis-resnet-cifar | ResNet18: CIFAR-10 입력과 출력에 맞춘 수정 | easy | `2. Vision/example_ipython/01_ResNet18_CIFAR10.ipynb` · 셀 9 · `build_resnet18_for_cifar10` |
| Vision | core-vis-down | U-Net Down: 해상도 축소 후 특징 추출 | easy | `2. Vision/example_ipython/04_Unet.ipynb` · 셀 7 · `Down.forward` |
| Vision | core-vis-dataset | CIFAR10: 학습·평가 분할과 전처리 연결 | easy | `2. Vision/example_ipython/01_ResNet18_CIFAR10.ipynb` · 셀 6 · `연속 코드 발췌` |
| Vision | core-vis-transformer | ViT Encoder: Attention·FFN의 연속 잔차 갱신 | medium | `2. Vision/example_ipython/02_ViT_CIFAR10.ipynb` · 셀 9 · `Transformer.forward` |
| Vision | core-vis-patch-build | ViT 구성: 이미지 패치와 분류 토큰 | medium | `2. Vision/example_ipython/02_ViT_CIFAR10.ipynb` · 셀 11 · `ViT` |
| Vision | core-vis-doubleconv | U-Net 구성: 해상도를 보존하는 두 Convolution | medium | `2. Vision/example_ipython/04_Unet.ipynb` · 셀 7 · `DoubleConv` |
| Vision | core-vis-vit | ViT: 이미지 패치에서 분류 출력까지 | hard | `2. Vision/example_ipython/02_ViT_CIFAR10.ipynb` · 셀 11 · `ViT.forward` |
| Vision | core-vis-unet | U-Net: Encoder·Skip·Decoder의 전체 연결 | hard | `2. Vision/example_ipython/04_Unet.ipynb` · 셀 7 · `UNet.forward` |
| Vision | core-vis-up | U-Net 디코더: 해상도 복원과 skip 특징 결합 | hard | `2. Vision/example_ipython/04_Unet.ipynb` · 셀 7 · `Up.forward` |
| Vision | core-vis-attention | ViT Attention: QK 점수와 V 가중합 | hard | `2. Vision/example_ipython/02_ViT_CIFAR10.ipynb` · 셀 9 · `Attention.forward` |
| Vision | core-vis-unet-build | U-Net 구성: Encoder·Decoder 채널과 출력 Mask | hard | `2. Vision/example_ipython/04_Unet.ipynb` · 셀 7 · `UNet` |
| On-device | core-od-quantization | 가중치 양자화·역양자화의 전체 과정 | easy | `3. On-device/example_ipython/2. Advanced Quantization Lab_answer.ipynb` · 셀 18 · `pseudo_quantize_tensor` |
| On-device | core-od-scale-zero | 선형 양자화: Scale과 Zero Point 계산 | easy | `3. On-device/example_ipython/1. Basic Quantization Lab_answer.ipynb` · 셀 34 · `get_quantization_scale_and_zero_point` |
| On-device | core-od-cache-memory | KV Cache: 메모리 사용량과 GQA Head 수 | easy | `3. On-device/example_ipython/4_[Colab]_KV_Cache_Pruning_answer.ipynb` · 셀 10 · `kv_cache_bytes` |
| On-device | core-od-perchannel | 레이어의 출력 채널별 가중치 양자화 | medium | `3. On-device/example_ipython/1. Basic Quantization Lab_answer.ipynb` · 셀 44 · `linear_quantize_weight_per_channel` |
| On-device | core-od-kmeans | K-means 양자화: Codebook으로 Tensor 복원 | medium | `3. On-device/example_ipython/1. Basic Quantization Lab_answer.ipynb` · 셀 86 · `k_means_quantize` |
| On-device | core-od-qat-codebook | QAT: 학습된 가중치로 Centroid 갱신 | medium | `3. On-device/example_ipython/1. Basic Quantization Lab_answer.ipynb` · 셀 97 · `update_codebook` |
| On-device | core-od-streaming | StreamingLLM: Sink와 최근 토큰 보존 | medium | `3. On-device/example_ipython/4_[Colab]_KV_Cache_Pruning_answer.ipynb` · 셀 18 · `streaming_score` |
| On-device | core-od-h2o-norm | H2O-Norm: 누적 Attention의 위치 편향 보정 | medium | `3. On-device/example_ipython/4_[Colab]_KV_Cache_Pruning_answer.ipynb` · 셀 24 · `h2o_norm_score` |
| On-device | core-od-draft | Speculative Draft: 토큰과 Cache의 연속 갱신 | medium | `3. On-device/example_ipython/5_[Colab]Speculative_Decoding_Answer.ipynb` · 셀 8 · `draft_tokens` |
| On-device | core-od-verify | Target 검증: 확정 토큰과 Draft의 위치 대응 | medium | `3. On-device/example_ipython/5_[Colab]Speculative_Decoding_Answer.ipynb` · 셀 8 · `verify_draft` |
| On-device | core-od-pruning | KV Cache Pruning: 중요도 선택과 메모리 축소 | hard | `3. On-device/example_ipython/4_[Colab]_KV_Cache_Pruning_answer.ipynb` · 셀 16 · `연속 코드 발췌` / `3. On-device/example_ipython/4_[Colab]_KV_Cache_Pruning_answer.ipynb` · 셀 21 |
| On-device | core-od-speculative | Speculative Decoding: Target 검증과 수락 알고리즘 | hard | `3. On-device/example_ipython/5_[Colab]Speculative_Decoding_Answer.ipynb` · 셀 9 · `accept_draft` |
| On-device | core-od-distillation | 지식증류: Teacher 지식과 정답으로 Student 학습 | hard | `3. On-device/example_ipython/3. Knowledge Distillation_answer.ipynb` · 셀 27 · `연속 코드 발췌` |
| On-device | core-od-linear | 양자화 Linear: 정수 누산과 출력 재양자화 | hard | `3. On-device/example_ipython/1. Basic Quantization Lab_answer.ipynb` · 셀 56 · `quantized_linear` |
| On-device | core-od-smooth | SmoothQuant: 활성값·가중치 범위의 균형 | hard | `3. On-device/example_ipython/2. Advanced Quantization Lab_answer.ipynb` · 셀 56 · `smooth_ln_fcs` |
| On-device | core-od-awq | AWQ: 중요 채널 보호와 양자화 후 스케일 복원 | hard | `3. On-device/example_ipython/2. Advanced Quantization Lab_answer.ipynb` · 셀 28 · `pseudo_quantize_model_weight_scaleup` |
| On-device | core-od-rotation | 회전 양자화: 입력·출력 Projection의 좌표 변환 | hard | `3. On-device/example_ipython/2. Advanced Quantization Lab_answer.ipynb` · 셀 68 · `rotate_model_weight` |
| On-device | core-od-kd-cosine | Cosine KD: 숨은 표현과 정답으로 Student 학습 | hard | `3. On-device/example_ipython/3. Knowledge Distillation_answer.ipynb` · 셀 40 · `연속 코드 발췌` |
| On-device | core-od-kd-hint | Hint KD: Feature Map 회귀와 정답 학습 | hard | `3. On-device/example_ipython/3. Knowledge Distillation_answer.ipynb` · 셀 53 · `연속 코드 발췌` |
| On-device | core-od-snapkv | SnapKV: 관측 Window와 이웃 중요도 결합 | hard | `3. On-device/example_ipython/4_[Colab]_KV_Cache_Pruning_answer.ipynb` · 셀 27 · `snapkv_score` |
| On-device | core-od-dflash | DFlash: Hidden Block에서 병렬 후보 생성 | hard | `3. On-device/example_ipython/5_[Colab]Speculative_Decoding_Answer.ipynb` · 셀 30 · `parallel_block_draft` |
| On-device | core-od-awq-scales | AWQ: 활성값 기반 채널별 Scale 적용 | hard | `3. On-device/example_ipython/2. Advanced Quantization Lab_answer.ipynb` · 셀 35 · `연속 코드 발췌` |
| Data | core-data-window | 시계열 데이터: 과거 구간과 미래 정답 구성 | easy | `4. Data/3_4-ts-practice/ts_solution.ipynb` · 셀 16 · `convert_data_into_tensors` |
| Data | core-data-evaluation | 시계열 평가: 추론 결과와 RMSE·MAPE | easy | `4. Data/3_4-ts-practice/ts_solution.ipynb` · 셀 28 · `test` |
| Data | core-data-time-split | 시계열 분할: 과거 학습과 미래 평가 | easy | `4. Data/3_4-ts-practice/ts_solution.ipynb` · 셀 12 · `연속 코드 발췌` |
| Data | core-data-mape | 시계열 평가: 절대 오차와 상대 오차 | easy | `4. Data/3_4-ts-practice/ts_solution.ipynb` · 셀 28 · `test` |
| Data | core-data-lstm | 시계열 예측 모델: LSTM 구성과 텐서 흐름 | medium | `4. Data/3_4-ts-practice/ts_solution.ipynb` · 셀 22 · `LSTMModel` |
| Data | core-data-conv | Conv1D 시계열 모델: 시간·채널 축 연결 | medium | `4. Data/3_4-ts-practice/ts_solution.ipynb` · 셀 30 · `Conv1DModel.forward` |
| Data | core-data-ncf | NCF: 사용자·아이템 Embedding 결합 | medium | `4. Data/7_8-recsys-practice/RecSys_NCF.ipynb` · 셀 11 · `Neural_Collaborative_Filtering.forward` |
| Data | core-data-rnn | RNN: 시점별 Hidden과 예측 연결 | medium | `4. Data/3_4-ts-practice/ts_solution.ipynb` · 셀 35 · `RNNModel` |
| Data | core-data-recsys-split | 추천 데이터: 연속 ID와 평점 분포 유지 | medium | `4. Data/7_8-recsys-practice/RecSys_NCF.ipynb` · 셀 9 · `연속 코드 발췌` |
| Data | core-data-ncf-train | NCF 학습: 평점 Shape와 회귀 손실 | medium | `4. Data/7_8-recsys-practice/RecSys_NCF.ipynb` · 셀 15 · `연속 코드 발췌` |
| Data | core-data-graph-edges | 추천 Graph: 사용자와 아이템 Node 분리 | medium | `4. Data/7_8-recsys-practice/RecSys_GCF_sol.ipynb` · 셀 10 · `create_edge_index` |
| Data | core-data-graph-split | 추천 Graph: Train·Validation·Test Edge 분리 | medium | `4. Data/7_8-recsys-practice/RecSys_GCF_sol.ipynb` · 셀 11 · `연속 코드 발췌` |
| Data | core-data-bpr | 추천 모델: 선호 순위를 학습하는 BPR 손실 | hard | `4. Data/7_8-recsys-practice/RecSys_GCF_sol.ipynb` · 셀 15 · `NGCF.bpr_loss` |
| Data | core-data-encoder-decoder | Encoder–Decoder: 상태 전달과 자기회귀 예측 | hard | `4. Data/3_4-ts-practice/ts_solution.ipynb` · 셀 44 · `RNNRNN.forward` |
| Data | core-data-ngcf-layer | NGCF: 이웃 메시지의 Degree 정규화 | hard | `4. Data/7_8-recsys-practice/RecSys_GCF_sol.ipynb` · 셀 14 · `NGCFLayer.forward` |
| Data | core-data-ranking-metrics | 추천 평가: Recall·Precision·NDCG | hard | `4. Data/7_8-recsys-practice/RecSys_GCF_sol.ipynb` · 셀 17 · `evaluate` |
| Data | core-data-ngcf-representation | NGCF: 여러 깊이의 표현과 사용자·아이템 분리 | hard | `4. Data/7_8-recsys-practice/RecSys_GCF_sol.ipynb` · 셀 15 · `NGCF.forward` |
| Data | core-data-bpr-train | NGCF 학습: 사용자·Positive·Negative와 BPR 연결 | hard | `4. Data/7_8-recsys-practice/RecSys_GCF_sol.ipynb` · 셀 19 · `연속 코드 발췌` |
| Data | core-data-ndcg | 추천 평가: 순위 할인과 이상적인 DCG | hard | `4. Data/7_8-recsys-practice/RecSys_GCF_sol.ipynb` · 셀 17 · `evaluate` |
| RAG | core-rag-pipeline | 기본 RAG: 검색 근거를 Reader 응답에 연결 | easy | `5. RAG/실습자료_Colab/2일차/code/2. Task_1.ipynb` · 셀 34 · `RAG.inference` |
| RAG | core-rag-synthesis | LlamaIndex: 검색 노드와 응답 합성기 연결 | easy | `5. RAG/실습자료_Colab/1일차/code/1. Llama_index.ipynb` · 셀 75 · `StandardQueryEngine.custom_query` |
| RAG | core-rag-crag-score | RAG 평가: 정답·보류·환각 점수 | easy | `5. RAG/실습자료_Colab/2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb` · 셀 37 · `record_result` |
| RAG | core-rag-mcp | MCP RAG: 비동기 검색과 생성의 전체 흐름 | medium | `5. RAG/실습자료_Colab/2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb` · 셀 79 · `RAGwithMCP.inference` |
| RAG | core-rag-llamaindex | LlamaIndex: 문서 인덱싱에서 검색까지 | medium | `5. RAG/실습자료_Colab/2일차/code/2. Task_1.ipynb` · 셀 21 · `연속 코드 발췌` |
| RAG | core-rag-hybrid | Graph RAG: 질의에 맞는 근거 선택과 생성 | medium | `5. RAG/실습자료_Colab/2일차/code/3. Task_2.ipynb` · 셀 43 · `RAGWithSRKG.inference` |
| RAG | core-rag-context | Custom QueryEngine: 검색 근거에서 생성 프롬프트까지 | medium | `5. RAG/실습자료_Colab/1일차/code/1. Llama_index.ipynb` · 셀 80 · `OurCustomQueryEngine.custom_query` |
| RAG | core-rag-reader-prompt | RAG Reader: 검색 근거만 사용하는 Prompt | medium | `5. RAG/실습자료_Colab/2일차/code/2. Task_1.ipynb` · 셀 30 · `Reader.prompt_generator` |
| RAG | core-rag-load-chunk | RAG 준비: 문서 로딩에서 겹치는 Chunk까지 | medium | `5. RAG/실습자료_Colab/1일차/code/1. Llama_index.ipynb` · 셀 21 · `연속 코드 발췌` / `5. RAG/실습자료_Colab/1일차/code/1. Llama_index.ipynb` · 셀 12 |
| RAG | core-rag-embedding-call | Embedding: 단일 질의와 문서 배치의 공통 벡터 공간 | medium | `5. RAG/실습자료_Colab/2일차/code/2. Task_1.ipynb` · 셀 17 · `BaseRetriever.embed_text` |
| RAG | core-rag-html | 검색 문서 전처리: HTML에서 본문 추출 | medium | `5. RAG/실습자료_Colab/2일차/code/3. Task_2.ipynb` · 셀 42 · `parse_htmls` |
| RAG | core-rag-reader-generate | Reader 생성: 근거 Prompt에서 답변 추출 | medium | `5. RAG/실습자료_Colab/2일차/code/2. Task_1.ipynb` · 셀 30 · `Reader.generate_response` |
| RAG | core-rag-mcp-discovery | MCP 연결: Client에서 Tool·Resource 조회까지 | medium | `5. RAG/실습자료_Colab/2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb` · 셀 67 · `연속 코드 발췌` / `5. RAG/실습자료_Colab/2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb` · 셀 66 |
| RAG | core-rag-grade | RAG 평가: 정확한 답·허용 답·보류·오답 구분 | medium | `5. RAG/실습자료_Colab/2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb` · 셀 37 · `grade_prediction` |
| RAG | core-rag-mcp-time | MCP RAG: 질의 시각을 포함한 검색 요청 | medium | `5. RAG/실습자료_Colab/2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb` · 셀 79 · `RAGwithMCP.retrieve` |
| RAG | core-rag-retriever | Retriever: 문서·청크·임베딩·유사도 검색 | hard | `5. RAG/실습자료_Colab/2일차/code/2. Task_1.ipynb` · 셀 17 · `BaseRetriever.retrieve` |
| RAG | core-rag-mcp-agent | MCP Agent: 도구·LLM·대화 Context 구성 | hard | `5. RAG/실습자료_Colab/2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb` · 셀 70 · `KGQueryEngineWithMCP.init_agent` |
| RAG | core-rag-mcp-query | MCP 실행: Context 전달과 비동기 답변 수신 | hard | `5. RAG/실습자료_Colab/2일차/code/4_RAG_framework_evaluation_with_MCP.ipynb` · 셀 70 · `KGQueryEngineWithMCP.query` |
