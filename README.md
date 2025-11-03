# Emotion-Engine-Demo
#  AI NPC 감정 엔진 SDK (v1.0)

8년간 독자적으로 연구한 AI 인지 아키텍처(EIDOS)의 핵심 모듈인 '감정 엔진'을 분리하여 SDK로 만들었습니다.

이 엔진은 단순한 '감정 상태'가 아니라, 감정의 '관성(Momentum)'과 '상호작용(Dynamics)'을 시뮬레이션하여 살아있는 듯한 NPC 반응을 구현합니다.

---

###  실시간 웹 데모

**[➡️ 지금 바로 데모 체험하기 (Streamlit Cloud)]**
*(여기에 배포한 Streamlit URL을 넣어주세요: 예: https://npc-demo.streamlit.app)*

---

###  데모 시연 (GIF/스크린샷)

*(여기에 `demo_npc.py`를 실행하고 '선물하기'/'모욕하기' 버튼을 눌렀을 때 NPC의 반응이 변하는 장면을 GIF로 녹화하여 삽입하세요.)*

(이미지 예시)
![NPC Demo Screenshot](httpsMultiModalEncoder/placeholder.gif)

---

### 핵심 기능 (Why this Engine?)

기존의 챗봇이나 NPC는 감정이 스위치처럼 켜지고 꺼집니다. 이 엔진은 두 가지 독창적인 EIDOS 로직을 사용하여 이 문제를 해결합니다.

**1. 감정 관성 (EmotionMomentum)**
감정은 즉시 최고조에 달했다가 0으로 떨어지지 않습니다. 이 엔진은 감정의 '관성'을 구현하여, 자극(Delta)이 멈춘 후에도 감정이 서서히 식도록 시뮬레이션합니다. (예: `base_resistance`, `momentum_factor`)

**2. 감정 동역학 (EmotionDynamics) [⭐ 핵심]**
감정은 독립적이지 않습니다. 이 엔진은 12차원의 감정이 서로 상호작용하도록 설계되었습니다. (`interaction_matrix`)

* **(예: 신뢰와 분노)** NPC가 플레이어를 깊게 '신뢰(Trust)'(6)하는 상태라면, '분노(Anger)'(2)를 유발하는 이벤트가 발생해도 '신뢰'가 '분노'를 억제합니다.
* **(예: 착잡함)** '기쁨(Joy)'(0)과 '슬픔(Sadness)'(1)이 동시에 높으면 '착잡함(bittersweet)'이라는 복합 감정으로 탐지됩니다.

---

### 빠른 시작 (Quickstart)

`pip install numpy` 외에 특별한 의존성이 없습니다.

```python
import numpy as np
from emotion_engine_sdk import EmotionEngineSDK, EMOTION_DIM

# 1. SDK 로드
sdk = EmotionEngineSDK()

print(f"초기 감정: {sdk.get_current_emotions()}")

# 2. '감정 변화(Delta)' 이벤트 생성
# (예: '기쁨' +0.5, '신뢰' +0.3 자극)
delta_event = np.zeros(EMOTION_DIM)
delta_event[0] = 0.5 # 기쁨
delta_event[6] = 0.3 # 신뢰

# 3. SDK에 이벤트 주입
# (엔진이 동역학과 관성을 자동 계산)
sdk.process_event(delta_event)

# 4. 결과 확인
final_emotions = sdk.state.activations
print(f"이벤트 후 감정: {final_emotions}")

# 5. 복합 감정 분석
purity, complex_states = sdk.analyze_complex_emotions()
if complex_states:
    print(f"복합 감정 발생: {complex_states}")



------------설치방법-------------------
# 1. 이 저장소를 클론합니다.
git clone (여기에 이 GitHub 저장소 URL)

# 2. 필요한 라이브러리를 설치합니다.
pip install -r requirements.txt

# 3. NPC 데모를 로컬에서 실행합니다.
streamlit run demo_npc.py
-------------------------------------

<피드백>
이 엔진은 EIDOS 아키텍처의 첫 번째 공개 모듈입니다. 사용해 보시고 어떤 피드백(버그, 기능 제안)이든 편하게 GitHub Issue로 남겨주세요!
