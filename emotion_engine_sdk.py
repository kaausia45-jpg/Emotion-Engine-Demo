import numpy as np
from typing import Dict, Tuple, List, Any, Optional

# --- EIDOS의 상수 (설계도에서 복사) ---
EMOTION_DIM = 12
EMOTION_MIN = 0.0
EMOTION_MAX = 2.0
EMOTION_MAP = { 
    0: "기쁨", 1: "슬픔", 2: "분노", 3: "공포", 4: "놀람", 
    5: "혐오", 6: "신뢰", 7: "기대", 8: "수치심", 9: "자부심", 
    10: "호기심", 11: "지루함" 
}


class EmotionState:
    # ... (v11.1과 동일) ...
    def __init__(self): self.activations = np.zeros(EMOTION_DIM); print("[Emotion] EmotionState (v9.1)")
    def get_vector(self) -> np.ndarray: return self.activations.reshape(1, -1)
    def update(self, new_activations: np.ndarray): self.activations = np.clip(new_activations, EMOTION_MIN, EMOTION_MAX)
    def to_dict(self) -> List[float]: return self.activations.tolist()
    def from_dict(self, data: List[float]): self.activations = np.array(data)

class EmotionDynamics:
    # ... (v11.1과 동일) ...
    def __init__(self):
        base_rates = {'기쁨': 0.90, '슬픔': 0.95, '분노': 0.85, '공포': 0.98, '놀람': 0.80, '혐오': 0.96, '신뢰': 0.97, '기대': 0.93, '수치심': 0.94, '자부심': 0.96, '호기심': 0.92, '지루함': 0.90}
        self.base_decay_vector = np.ones(EMOTION_DIM);
        for i, name in EMOTION_MAP.items():
             if name in base_rates: self.base_decay_vector[i] = base_rates[name]
        self.interaction_matrix: Dict[Tuple[int, int], float] = {(0, 1): -0.8, (1, 0): -0.6, (2, 3): -0.6, (3, 2): -0.3, (6, 5): -0.7, (5, 6): -0.8, (0, 6): 0.5, (6, 0): 0.3, (3, 4): 0.6, (4, 3): 0.4, (7, 0): 0.4, (0, 7): 0.2,}
        print(f"[Emotion] EmotionDynamics (v9.2)"); self.fear_index = 3
    def apply_dynamics(self, current_activations: np.ndarray, emotion_delta: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        decay_vector = self.base_decay_vector.copy()
        if context and context.get('is_traumatic'): decay_vector[self.fear_index] = 0.999; print("![EmotionDynamics] Trauma Context")
        decayed_activations = current_activations * decay_vector; new_activations = decayed_activations + emotion_delta
        interacted_activations = new_activations.copy(); interaction_threshold = 0.5
        for (idx_a, idx_b), strength in self.interaction_matrix.items():
            if new_activations[idx_a] > interaction_threshold:
                if strength < 0: interacted_activations[idx_b] *= (1 + strength)
                else: interacted_activations[idx_b] *= (1 + strength)
        return np.maximum(interacted_activations, 0.0)

# eidos_v4_0_core.py (또는 최신 코어 파일) 내 EmotionMemory 클래스 수정
class EmotionMemory:
    def __init__(self, max_sources_per_emotion=5):
        self.sources: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(EMOTION_DIM)}
        self.max_sources = max_sources_per_emotion
        self.current_time = 0
        print("[Emotion] EmotionMemory (v9.1 / Patched)") # 버전명에 패치 표시

    def tick(self): self.current_time += 1

    def record_cause(self, emotion_index: int, cause: str, intensity: float):
        if abs(intensity) < 0.01: return

        # [오류 수정] .get()을 사용하여 안전하게 리스트 가져오기
        source_list = self.sources.get(emotion_index)

        # [오류 수정] 만약 해당 감정 인덱스에 대한 리스트가 없으면 새로 생성
        if source_list is None:
            self.sources[emotion_index] = []
            source_list = self.sources[emotion_index] # 새로 생성된 리스트를 source_list에 할당

        found = False
        # 이제 source_list는 항상 할당되어 있음
        for source in source_list:
            if source['cause'] == cause:
                # 기존 원인이면 강도 업데이트 및 타임스탬프 갱신
                source['intensity'] = max(source['intensity'], intensity)
                # [오타 수정 가능성] self.current.time -> self.current_time
                source['timestamp'] = self.current_time
                found = True
                break # 이미 찾았으므로 루프 종료

        # 기존 원인이 아니면 새로 추가
        if not found:
            # [오타 수정 가능성] self.current.time -> self.current_time
            source_list.append({ 'cause': cause, 'intensity': intensity, 'timestamp': self.current_time })

            # 리스트 크기 제한 유지
            if len(source_list) > self.max_sources:
                source_list.sort(key=lambda x: x['timestamp'], reverse=True)
                # self.sources[emotion_index] = source_list[:self.max_sources] # 잘라낸 리스트로 딕셔너리 업데이트
                # 위 라인은 아래와 같이 직접 슬라이싱 할당으로 변경 가능
                self.sources[emotion_index][:] = source_list[:self.max_sources]


    def record_causes_from_delta(self, delta_vector: np.ndarray, cause_prefix: str):
        # ... (기존과 동일) ...
        for i, intensity in enumerate(delta_vector):
            if abs(intensity) > 0.1: self.record_cause(i, cause_prefix, intensity)

    def get_explanation(self, emotion_index: int) -> str:
        # ... (기존과 동일) ...
        source_list = self.sources.get(emotion_index, [])
        if not source_list: return "특별한 원인 없음"
        main_source = max(source_list, key=lambda x: abs(x['intensity']))
        return f"{main_source['cause']} (강도 {main_source['intensity']:.2f})"

    def get_primary_cause_string(self, current_activations: np.ndarray) -> str:
        # ... (기존과 동일) ...
        try:
            primary_emotion_index = int(np.argmax(current_activations))
            primary_emotion_name = EMOTION_MAP.get(primary_emotion_index, "?")
            primary_cause = self.get_explanation(primary_emotion_index)
            return f"주요 감정 '{primary_emotion_name}' 원인: '{primary_cause}'."
        except Exception: return "주요 감정 원인 분석 불가."

# --- EidosCore 클래스 등 나머지 코드는 그대로 유지 ---

class EmotionMomentum:
    # ... (v11.1과 동일) ...
    def __init__(self, emotion_dim=EMOTION_DIM, momentum_factor=0.3, base_resistance=0.4):
        self.emotion_dim = emotion_dim; self.momentum_vector = np.zeros(emotion_dim); self.momentum_factor = momentum_factor; self.base_resistance = base_resistance
        self.fear_index = 3; self.resistance_map = { self.fear_index: {'increase': 0.1, 'decrease': 0.5} }; print("[Emotion] EmotionMomentum (v9.2)")
    def _compute_resistance(self, emotion_index: int, change_magnitude: float) -> float:
        config = self.resistance_map.get(emotion_index)
        if config: res = config['increase'] if change_magnitude > 0 else config['decrease']
        else: res = self.base_resistance + (abs(change_magnitude) * 0.2)
        return np.clip(res, 0.0, 0.9)
    def apply(self, current_activations: np.ndarray, target_activations: np.ndarray) -> np.ndarray:
        target_change = target_activations - current_activations; actual_change = np.zeros(self.emotion_dim)
        for i in range(self.emotion_dim):
            change = target_change[i];
            if abs(change) < 0.01: continue
            resistance = self._compute_resistance(i, change); resisted_change = change * (1 - resistance)
            momentum_effect = self.momentum_vector[i] * self.momentum_factor; actual_change[i] = resisted_change + momentum_effect
        self.momentum_vector = (self.momentum_vector * 0.5) + (actual_change * 0.5); final_activations = current_activations + actual_change
        return final_activations

class ComplexEmotionMonitor:
    # ... (v11.1과 동일) ...
    def __init__(self):
        self.COMPLEX_EMOTION_MAP = {(0, 1): "착잡함/bittersweet", (0, 6): "애정/affection", (3, 4): "경외/awe", (0, 7): "낙관/optimism", (2, 5): "경멸/contempt", (1, 8): "자책/self-blame",}
        self.THRESHOLD = 0.7; print("[Emotion] ComplexEmotionMonitor (v9.3)")
    def analyze_state(self, emotion_vector: np.ndarray) -> Tuple[float, Dict[str, float]]:
        activations = emotion_vector; sum_vals = np.sum(activations);
        if sum_vals == 0: return 1.0, {}
        max_val = np.max(activations); purity = max_val / sum_vals; detected_states = {}
        for (idx_a, idx_b), name in self.COMPLEX_EMOTION_MAP.items():
            val_a = activations[idx_a]; val_b = activations[idx_b]
            if val_a > self.THRESHOLD and val_b > self.THRESHOLD: strength = (val_a + val_b) / 2.0; detected_states[name] = strength
        return purity, detected_states

class EmotionEngineSDK:
    """
    이벤트 입력에 따라 동적 감정을 시뮬레이션하는 SDK입니다.
    EIDOS Core와 무관하게 독립적으로 작동합니다.
    """
    def __init__(self):
        # 2단계에서 복사한 부품들을 조립합니다.
        self.state = EmotionState()
        self.dynamics = EmotionDynamics()
        self.momentum = EmotionMomentum()
        self.monitor = ComplexEmotionMonitor()
        print("✅ Emotion Engine SDK (v1.0) Loaded.")

    def get_current_emotions(self) -> np.ndarray:
        """ 현재 감정 벡터 (1, 12)를 반환합니다. """
        return self.state.get_vector()

    def process_event(self, 
                      emotion_delta: np.ndarray, 
                      context: Optional[Dict[str, Any]] = None
                     ) -> np.ndarray:
        """
        [핵심 기능]
        외부에서 계산된 '감정 변화량(delta)'을 받아,
        동역학과 관성을 적용한 '최종 감정 상태'를 반환합니다.
        
        (이 로직은 EIDOS process_input의 감정 처리 로직을 '참고'하여 재구성)
        """
        
        # 1. 현재 상태 백업
        prev_activations = self.state.activations.copy()

        # 2. 감정 동역학 적용 (설계도 참고)
        target_activations = self.dynamics.apply_dynamics(
            prev_activations, 
            emotion_delta, 
            context=context
        )
        
        # 3. 감정 관성 적용 (설계도 참고)
        final_activations = self.momentum.apply(
            prev_activations, 
            target_activations
        )
        
        # 4. 상태 업데이트
        self.state.update(final_activations)
        
        return self.state.activations

    def analyze_complex_emotions(self) -> Tuple[float, Dict[str, float]]:
        """ 복합 감정 상태를 분석합니다. (설계도 참고) """
        return self.monitor.analyze_state(self.state.activations)

# --- 고객 사용 예시 (테스트 코드) ---
if __name__ == "__main__":
    sdk = EmotionEngineSDK()
    
    print(f"초기 감정: {sdk.get_current_emotions()}")
    
    # 가상의 이벤트 발생 (예: '기쁨' +0.5, '슬픔' +0.3)
    delta_1 = np.zeros(EMOTION_DIM)
    delta_1[0] = 0.5 # 기쁨
    delta_1[1] = 0.3 # 슬픔
    
    sdk.process_event(delta_1)
    
    print(f"이벤트 1 후: {sdk.get_current_emotions()}")
    
    purity, complex_states = sdk.analyze_complex_emotions()
    print(f"복합 감정: {complex_states} (순도: {purity:.2f})")
