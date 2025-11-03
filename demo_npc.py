# demo_npc.py
import streamlit as st
import numpy as np
import pandas as pd
# 방금 만든 SDK 클래스를 임포트합니다.
from emotion_engine_sdk import EmotionEngineSDK, EMOTION_DIM, EMOTION_MAP

# --- 1. SDK 초기화 ---
if 'sdk' not in st.session_state:
    st.session_state.sdk = EmotionEngineSDK() #
    # (EIDOS 설계도 참고) NPC의 기본 성격: 신뢰(6), 호기심(10)이 높음
    base_personality = np.zeros(EMOTION_DIM)
    base_personality[6] = 0.5 # 신뢰
    base_personality[10] = 0.3 # 호기심
    st.session_state.sdk.state.update(base_personality)

sdk = st.session_state.sdk

# --- 2. [핵심] 감정 상태를 '대화'로 번역하는 함수 ---
def generate_dialogue(activations: np.ndarray) -> (str, str):
    face = "😐"
    
    # 12개 감정 중 가장 높은 감정 찾기
    dominant_emotion_idx = np.argmax(activations)
    dominant_value = activations[dominant_emotion_idx]
    
    # EIDOS의 복합 감정 로직 흉내내기
    joy = activations[0]
    sadness = activations[1]
    anger = activations[2]
    trust = activations[6]

    if joy > 0.7 and trust > 0.5:
        face = "🥰"
        dialogue = f"와! 정말 고마워요! 당신은 역시 믿을 수 있는 분이에요! (기쁨: {joy:.2f}, 신뢰: {trust:.2f})"
    elif anger > 0.7 and trust < 0.3:
        face = "🤬"
        dialogue = f"...뭐라고요? 지금 날 무시하는 거예요? (분노: {anger:.2f}, 신뢰: {trust:.2f})"
    elif anger > 0.6 and trust > 0.6:
        face = "😠"
        dialogue = f"기분 나쁜 말이네요... 하지만 당신이 한 말이니까 뭔가 이유가 있겠죠. (분노: {anger:.2f}, 신뢰: {trust:.2f})"
    elif sadness > 0.8:
        face = "😭"
        dialogue = f"너무 슬퍼요... (슬픔: {sadness:.2f})"
    elif dominant_value < 0.2:
        face = "😐"
        dialogue = "(NPC는 특별한 반응이 없다.)"
    else:
        # 기타 단일 감정
        dominant_emotion_name = EMOTION_MAP.get(dominant_emotion_idx, "??")
        if dominant_emotion_name == "기쁨":
            face = "😄"
            dialogue = "기분 좋은 일이네요!"
        elif dominant_emotion_name == "분노":
            face = "😡"
            dialogue = "화가 나네요."
        elif dominant_emotion_name == "신뢰":
            face = "😊"
            dialogue = "당신을 믿어요."
        else:
            face = "🤔"
            dialogue = f"({dominant_emotion_name}..."
            
    return face, dialogue

# --- 3. GUI 레이아웃 ---
st.title("💖 AI NPC 감정 엔진 쇼케이스")
st.write("NPC에게 상호작용을 하여 감정 변화와 반응(대화)을 관찰하세요.")

# --- 4. 입력 (사이드바) ---
st.sidebar.header("🕹️ 플레이어 행동")

if st.sidebar.button("선물하기 🎁"):
    # (기쁨 +0.5, 신뢰 +0.3)
    delta_vec = np.array([0.5, 0, 0, 0, 0, 0, 0.3, 0.1, 0, 0.1, 0, 0])
    sdk.process_event(delta_vec) #

if st.sidebar.button("모욕하기 😠"):
    # (분노 +0.7, 슬픔 +0.2, 신뢰 -0.5)
    delta_vec = np.array([0, 0.2, 0.7, 0, 0, 0.1, -0.5, 0, 0.3, 0, 0, 0])
    sdk.process_event(delta_vec) #

if st.sidebar.button("도와주기 🙏"):
    # (기쁨 +0.3, 신뢰 +0.6, 자부심 +0.2)
    delta_vec = np.array([0.3, 0, 0, 0, 0, 0, 0.6, 0, 0, 0.2, 0, 0])
    sdk.process_event(delta_vec) #

if st.sidebar.button("시간이 흐름 (엔진 감쇠 테스트) ⏳"):
    # (아무 자극도 주지 않음 -> Dynamics의 감쇠 로직 테스트)
    delta_vec = np.zeros(EMOTION_DIM)
    sdk.process_event(delta_vec) #

# --- 5. 출력 (메인 화면) ---
current_emotions = sdk.state.activations.copy() #
face, dialogue = generate_dialogue(current_emotions)

st.header(f"NPC의 반응: {face}")
st.info(dialogue)

# --- 6. "Under the Hood" (고객이 원하면 볼 수 있는 계기판) ---
with st.expander("⚙️ 엔진 내부 상태 보기 (개발자용)"):
    st.subheader("📊 현재 감정 상태 (Activations)")
    
    chart_data = pd.DataFrame({
        "감정": [EMOTION_MAP.get(i, "?") for i in range(EMOTION_DIM)],
        "수준": current_emotions
    })
    
    # (경고 메시지 해결: width='stretch' 사용)
    st.bar_chart(chart_data.set_index("감정"), width='stretch')

    purity, complex_states = sdk.analyze_complex_emotions() #
    st.metric("감정 순도 (Purity)", f"{purity:.2f}")
    if complex_states:
        st.json(complex_states)
