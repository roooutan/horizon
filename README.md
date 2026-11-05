# horizon
Terminology-aware EN-KO academic translation LoRA
1. 모델 정보

항목

내용

Base Model

tencent/Hy-MT2-7B

초기 모델

Bomin CS Fine-tuning의 LoRA Adapter

학습 방법

LoRA Fine-tuning

추가 학습량

800 steps (약 1.7 epochs)

최종 모델

Terminology LoRA Adapter

용어 정책

glossary/찐막glossary_curated.json (172개 용어)
->최종 Adapter는 기존 CS LoRA에서 추가학습한 결과고, Base Model 전체 가중치 포함x.

->최종 학습 모델은 Releases에서 다운로드 가능. 파일명: terminology_adapter_backup.tar.gz
-> 주요파일:
src/prepare_terminology_sft_direct.py //SFT 데이터 전처리

src/train_terminology.py //Terminology LoRA 학습

src/verify_sft_data.py //데이터 분할 검증

src/evaluate_terminology.py //BLEU·chrF 및 용어 준수율 평가

src/evaluate_focused_terminology.py //전문용어 집중 평가

src/evaluate_markdown_preservation.py //Markdown 구조 보존 평가

glossary/찐막glossary_curated.json //최종 용어 정책

experiments/   //평가 결과 JSON·CSV

->성능 평가 결과:

기존 CS LoRA와 추가 학습한 Terminology LoRA를 동일 조건으로 비교.



평가 지표                     CS LoRA                                 Terminology LoRA

BLEU (KCI 624문장)             26.71                                        26.62


chrF (KCI 624문장)             35.00                                        35.06


전문용어 준수율                 68.42%                                       65.26%
(일반 지시문, 95건)


전문용어 준수율                 92.63%                                       93.68%
(Glossary 명시, 95건)


Markdown 구조 보존율            98.67%                                       98.67%
(75개 검사)

결과해석: 추가 ft만으로 뚜렷한 성능 향상 확인x. 전문용어 집중 평가와 Markdown 평가는 합성 테스트 기반. 자동 채점 결과는 일부 수동 검수 필요. 

->참고사항:
-추가학습은 완료. 현재 모델(Terminology LoRA) 기준으로 평가.
-최종 LoRA 가중치는 github release에 제공.
-평가 결과는 experiments/ 에서 확인 가능.
-실제 논문 적용 시 용어 정책 및 문맥상 번역 적절성 추가 확인 필요. 


---

# 전문용어 정책 구축 및 번역 모델 학습·평가

## 1. 전문용어 Glossary 구축

컴퓨터공학 학술 논문에서 전문용어의 원어 유지 및 한국어 번역 기준을 일관되게 적용하기 위해 전문용어 Glossary를 구축하였다.

**① 선행 작업: 데이터셋 정제 및 검증**

- KCI 기반 영어·한국어 논문 초록 데이터 약 10,059건의 품질을 점검하였다.
- JSON 형식 오류, 결측값, 중복 문장, 번역문 충돌, 원문·번역문 길이 불일치 등을 검사하였다.
- 중복 및 정렬 오류가 있는 데이터를 정제하고, 전문용어가 포함된 문장을 별도로 검토하였다.
- 정제된 데이터에서 학습 후보 4,972건을 확보하고, 용어 정책을 적용할 수 있는 문장과 문맥상 판단이 필요한 문장을 구분하였다.

**② Glossary 구성 기준**

전문용어별로 다음과 같은 정책을 적용하였다.

- **KEEP:** 번역하지 않고 영어 원어를 유지해야 하는 용어
- **TRANSLATE:** 정해진 한국어 전문용어로 번역하는 것이 적절한 용어
- **Context-sensitive:** 문맥에 따라 번역 여부나 표현을 판단해야 하는 용어
- **Auto-enforce:** 자동으로 번역 규칙을 강제하고 평가할 수 있는지 나타내는 속성

기존에 구축한 Glossary와 팀원들이 제공한 용어 정책을 통합하였으며, 중복되는 항목에서 정책이 충돌할 경우 팀의 합의된 정책을 우선하였다.

초기 Glossary 100개에 팀 데이터의 고유 용어 77개를 추가하여 177개로 병합한 뒤, 후속 검수를 거쳐 **최종 172개 용어(KEEP 45개, TRANSLATE 127개)**로 구성하였다.

최종 Glossary에는 용어별 번역 방식, 허용 번역어, 변형 표현, 약어, 문맥 의존 여부 등을 저장하였다.

**③ 용어 정책 설계 배경 및 문맥 기반 처리**

일부 전문용어는 한국어로 직역하면 원래의 기술적 의미가 흐려지거나 표현이 길고 낯설어져 오히려 이해하기 어려울 수 있다. 따라서 **원어 표기가 중요한 용어는 KEEP**, 한국어 표현이 널리 정착된 용어는 **TRANSLATE**로 분류하고, 그 외 일반적인 문장 서술은 자연스러운 한국어로 번역하는 것을 기본 방향으로 설정하였다.

또한 `tree`, `parent`, `child`처럼 컴퓨터공학에서는 특정 자료구조·노드 관계를 가리키지만 일상 문맥에서는 일반적인 사물을 뜻할 수도 있는 단어는 **단순 문자열 치환보다 문맥에 맞는 해석이 중요하다**고 보았다. 이러한 다의적 표현을 고려하여 문맥 의존 여부를 별도로 표시하고, 무조건적인 자동 강제를 제한하는 기준을 두었다. *(예시는 설계 취지 설명을 위한 것이며, 해당 단어 모두가 최종 Glossary에 등록되었다는 의미는 아니다.)*

## 2. 주요 코드 및 구현 내용

| 코드 | 주요 기능 |
|---|---|
| `term_detector.py` | 영문 문장에서 Glossary 전문용어 탐지 |
| `validate_terminology.py` | 용어 정책 적용 결과 및 탐지 정확성 검사 |
| `prepare_dataset.py` | 학술 논문 데이터 정제 및 학습 데이터 준비 |
| `sample_quality_review.py` | 데이터셋 품질 점검을 위한 표본 추출 |
| `audit_curated_terminology.py` | 용어 정책의 자동 강제 가능 여부 및 검토 대상 판정 |
| `prepare_terminology_sft_direct.py` | Glossary 규칙을 포함한 SFT 데이터 생성 |
| `verify_sft_data.py` | 학습·검증·테스트 데이터 구조 및 중복 검사 |
| `train_terminology.py` | LoRA 기반 전문용어 추가 학습 |
| `evaluate_terminology.py` | BLEU·chrF 및 용어 정책 평가 |
| `evaluate_focused_terminology.py` | 전문용어 집중 평가 |
| `evaluate_markdown_preservation.py` | Markdown 구조 보존 예비 평가 |

특히 용어 탐지기는 Glossary에 정의된 전문용어를 문장에서 식별하고, KEEP/TRANSLATE 정책에 따라 번역 지시문을 생성하도록 구현하였다.

## 3. 전문용어 번역 모델 추가 학습

**① 모델 구성**

- Base Model: `tencent/Hy-MT2-7B`
- 초기 모델: 팀원이 컴퓨터공학 논문 데이터로 Fine-tuning한 CS LoRA
- 학습 방식: LoRA 추가 학습, 4-bit NF4 양자화 활용
- 학습 환경: RunPod GPU
- 학습량: 800 steps

기존 CS LoRA를 기반으로 전문용어 정책이 반영된 데이터로 추가 학습하여 **Terminology LoRA**를 구축하였다.

**② 학습 데이터**

KCI 논문 기반 영어·한국어 병렬 데이터와 최종 Glossary를 활용하였다.

학습 데이터는 총 1,930건이며, 다음과 같이 구성하였다.

| 데이터 유형 | 건수 |
|---|---:|
| 전문용어 규칙 적용 데이터 | 243 |
| 문맥 의존 용어 포함 데이터 | 1,187 |
| 일반 번역 데이터 | 500 |
| **합계** | **1,930** |

학습·검증·테스트 데이터는 각각 1,930건, 620건, 624건으로 분리하고, 문서 및 문장 중복 여부를 검사하였다.

**③ 모델의 목적**

일반적인 영어→한국어 번역 능력을 유지하면서, 학술 논문에 등장하는 전문용어를 사전에 정의된 정책에 따라 일관되게 처리하도록 하는 것이 목적이다.

학습된 모델은 LoRA Adapter 형태로 저장하고 GitHub Release를 통해 팀원에게 전달하였다.

## 4. 기존 Terminology LoRA 성능평가

기존 CS LoRA와 추가 학습한 Terminology LoRA를 동일한 평가 데이터에서 비교하였다.

**① 전체 번역 품질**

KCI 테스트 데이터 624개를 사용하여 BLEU와 chrF를 측정하였다.

| 지표 | 기존 CS LoRA | Terminology LoRA |
|---|---:|---:|
| BLEU | 26.71 | 26.62 |
| chrF | 35.00 | 35.06 |

추가 학습 이후 일반 번역 품질에서 뚜렷한 개선은 확인되지 않았다.

**② 전문용어 정책 준수율**

Glossary에서 자동 평가가 가능한 용어 95개를 선정하고, 일반 번역 지시문과 Glossary 규칙을 명시한 지시문을 각각 적용하였다.

| 평가 조건 | 기존 CS LoRA | Terminology LoRA |
|---|---:|---:|
| 일반 지시문 | 68.42% | 65.26% |
| Glossary 명시 | 92.63% | 93.68% |

추가 학습 자체의 개선 효과는 제한적이었으나, Glossary 규칙을 명시적으로 제공하면 두 모델 모두 높은 정책 준수율을 보였다.

**③ Markdown 구조 보존 예비 평가**

기존 두 모델에 대해 합성 Markdown 문서 36개를 이용한 구조 보존 예비 평가를 진행하였다. 제목, 목록, 표, 코드, 수식 등의 구조 요소 75건을 검사한 결과 두 모델 모두 74건(98.67%)을 보존하였다.

이는 합성 데이터에 대한 예비 평가이며, 현재 팀의 최종 모델 및 실제 PDF 변환 시스템의 성능을 나타내는 수치는 아니다.

## 5. 최종 번역 시스템의 외부 모델 비교 평가 (진행 중)

팀에서 개발한 최종 Stage4 + Reranker 번역 시스템을 대상으로 외부 공개 번역 모델과 성능을 비교하고 있다.

**① 비교 대상**

- Meta M2M100-418M
- Meta NLLB-200 distilled 600M
- Tencent Hy-MT2-7B 원본 모델
- 우리 팀 최종 번역 시스템(Stage4 + Reranker)

**② 평가 데이터 및 방법**

- 평가 데이터: KCI 기반 논문 초록 624개
- BLEU: SacreBLEU의 한국어 `ko-mecab` 토크나이저 활용
- chrF: SacreBLEU의 문자 단위 평가
- 전문용어 정책 준수율: 최종 Glossary의 KEEP/TRANSLATE 기준
- 추가 분석: 숫자 보존율, 반복 출력, 비정상 번역 여부

각 모델의 번역 결과를 동일한 테스트 데이터 및 평가 코드로 채점하는 방식으로 진행하고 있다.

**③ 현재 확보된 결과**

| 모델 | BLEU | chrF | 전문용어 준수율 |
|---|---:|---:|---:|
| M2M100-418M | 8.97 | 18.20 | 54.55% |
| NLLB-200 600M | 13.93 | 23.60 | 75.76% |
| Hy-MT2-7B 원본 | 평가 중 | 평가 중 | 평가 중 |
| 팀 최종 시스템 | 28.23 | 36.69 | 98.48% |

팀 최종 시스템의 점수는 해당 모델 개발자가 제공한 평가 결과이며, 외부 모델 점수는 직접 실행하여 확보하였다. 최종 보고서에서는 평가 데이터와 채점 조건의 일치 여부를 추가 검증할 예정이다.

## 6. 최종 모델의 Markdown 보존 평가 계획 (미진행)

Markdown 형식의 영문 텍스트를 최종 번역 시스템에 입력하고, 번역 결과에서 원본 문서의 구조가 유지되는지를 평가할 예정이다.

주요 평가 항목은 제목 계층, 목록, 표 구조, 코드 블록, 수식, 링크 및 인용 형식 등이다.

기존 예비 평가 코드를 참고하되, 최종 시스템에 맞는 테스트 데이터와 채점 기준을 새로 확정할 예정이다.

실제 논문 PDF의 Markdown 변환 및 PDF 재출력 정확도는 Markdown 담당 팀의 파이프라인 완성 이후 별도로 검증한다.
