# 개인 맞춤형 복합 자극 기반 이명 치료 시스템

<div align="center">
  <svg width="800" height="600" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
    <!-- 시스템 전체 배경 -->
    <rect width="800" height="600" fill="#f8f9fa" rx="10" ry="10"/>
    
    <!-- 타이틀 -->
    <text x="400" y="40" font-family="Arial" font-size="24" text-anchor="middle" font-weight="bold" fill="#333">
      개인 맞춤형 복합 자극 기반 이명 치료 시스템
    </text>
    
    <!-- 다중 생체신호 분석부 -->
    <g transform="translate(50, 80)">
      <rect width="700" height="100" rx="5" ry="5" fill="#e3f2fd" stroke="#2196f3" stroke-width="2"/>
      <text x="350" y="30" font-family="Arial" font-size="18" text-anchor="middle" font-weight="bold" fill="#0d47a1">
        다중 생체신호 분석부 (100)
      </text>
      
      <!-- 모듈들 -->
      <g transform="translate(40, 50)">
        <rect width="180" height="30" rx="5" ry="5" fill="#bbdefb" stroke="#64b5f6" stroke-width="1"/>
        <text x="90" y="20" font-family="Arial" font-size="14" text-anchor="middle" fill="#333">
          고해상도 청력 검사 모듈
        </text>
      </g>
      
      <g transform="translate(260, 50)">
        <rect width="180" height="30" rx="5" ry="5" fill="#bbdefb" stroke="#64b5f6" stroke-width="1"/>
        <text x="90" y="20" font-family="Arial" font-size="14" text-anchor="middle" fill="#333">
          뇌파 분석 모듈
        </text>
      </g>
      
      <g transform="translate(480, 50)">
        <rect width="180" height="30" rx="5" ry="5" fill="#bbdefb" stroke="#64b5f6" stroke-width="1"/>
        <text x="90" y="20" font-family="Arial" font-size="14" text-anchor="middle" fill="#333">
          수면 다원검사 모듈
        </text>
      </g>
    </g>
    
    <!-- 자극 생성부 섹션 -->
    <g transform="translate(50, 190)">
      <!-- 청각 자극 생성부 -->
      <g transform="translate(0, 0)">
        <rect width="220" height="100" rx="5" ry="5" fill="#e8f5e9" stroke="#4caf50" stroke-width="2"/>
        <text x="110" y="30" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold" fill="#1b5e20">
          청각 자극 생성부 (200)
        </text>
        
        <g transform="translate(20, 50)">
          <rect width="180" height="40" rx="5" ry="5" fill="#c8e6c9" stroke="#81c784" stroke-width="1"/>
          <text x="90" y="15" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
            노치 필터 모듈
          </text>
          <text x="90" y="30" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
            주파수/진폭 변조 모듈
          </text>
        </g>
      </g>
      
      <!-- 시각 자극 생성부 -->
      <g transform="translate(240, 0)">
        <rect width="220" height="100" rx="5" ry="5" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
        <text x="110" y="30" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold" fill="#e65100">
          시각 자극 생성부 (300)
        </text>
        
        <g transform="translate(20, 50)">
          <rect width="180" height="40" rx="5" ry="5" fill="#ffe0b2" stroke="#ffb74d" stroke-width="1"/>
          <text x="90" y="15" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
            패턴 생성 모듈
          </text>
          <text x="90" y="30" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
            색상 최적화 모듈
          </text>
        </g>
      </g>
      
      <!-- 촉각 자극 생성부 -->
      <g transform="translate(480, 0)">
        <rect width="220" height="100" rx="5" ry="5" fill="#e1f5fe" stroke="#03a9f4" stroke-width="2"/>
        <text x="110" y="30" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold" fill="#01579b">
          촉각 자극 생성부 (400)
        </text>
        
        <g transform="translate(20, 50)">
          <rect width="180" height="40" rx="5" ry="5" fill="#b3e5fc" stroke="#4fc3f7" stroke-width="1"/>
          <text x="90" y="15" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
            진동 패턴 생성 모듈
          </text>
          <text x="90" y="30" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
            강도 조절 모듈
          </text>
        </g>
      </g>
    </g>
    
    <!-- 복합 자극 출력부 -->
    <g transform="translate(50, 300)">
      <rect width="700" height="80" rx="5" ry="5" fill="#f3e5f5" stroke="#9c27b0" stroke-width="2"/>
      <text x="350" y="30" font-family="Arial" font-size="18" text-anchor="middle" font-weight="bold" fill="#4a148c">
        복합 자극 출력부 (500)
      </text>
      
      <!-- 출력 모듈들 -->
      <g transform="translate(40, 45)">
        <rect width="180" height="25" rx="5" ry="5" fill="#e1bee7" stroke="#ba68c8" stroke-width="1"/>
        <text x="90" y="17" font-family="Arial" font-size="14" text-anchor="middle" fill="#333">
          청각 자극 출력 모듈
        </text>
      </g>
      
      <g transform="translate(260, 45)">
        <rect width="180" height="25" rx="5" ry="5" fill="#e1bee7" stroke="#ba68c8" stroke-width="1"/>
        <text x="90" y="17" font-family="Arial" font-size="14" text-anchor="middle" fill="#333">
          시각 자극 출력 모듈
        </text>
      </g>
      
      <g transform="translate(480, 45)">
        <rect width="180" height="25" rx="5" ry="5" fill="#e1bee7" stroke="#ba68c8" stroke-width="1"/>
        <text x="90" y="17" font-family="Arial" font-size="14" text-anchor="middle" fill="#333">
          촉각 자극 출력 모듈
        </text>
      </g>
    </g>
    
    <!-- 수면 관리부 -->
    <g transform="translate(50, 390)">
      <rect width="330" height="100" rx="5" ry="5" fill="#ffebee" stroke="#f44336" stroke-width="2"/>
      <text x="165" y="30" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold" fill="#b71c1c">
        수면 관리부 (600)
      </text>
      
      <g transform="translate(20, 50)">
        <rect width="290" height="40" rx="5" ry="5" fill="#ffcdd2" stroke="#ef9a9a" stroke-width="1"/>
        <text x="145" y="15" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
          수면 단계 감지 모듈
        </text>
        <text x="145" y="30" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
          자극 최적화 모듈
        </text>
      </g>
    </g>
    
    <!-- 적응형 제어부 -->
    <g transform="translate(420, 390)">
      <rect width="330" height="100" rx="5" ry="5" fill="#fffde7" stroke="#ffeb3b" stroke-width="2"/>
      <text x="165" y="30" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold" fill="#f57f17">
        적응형 제어부 (700)
      </text>
      
      <g transform="translate(20, 50)">
        <rect width="290" height="40" rx="5" ry="5" fill="#fff9c4" stroke="#fff59d" stroke-width="1"/>
        <text x="145" y="15" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
          반응 분석 모듈
        </text>
        <text x="145" y="30" font-family="Arial" font-size="12" text-anchor="middle" fill="#333">
          자극 최적화 모듈
        </text>
      </g>
    </g>
    
    <!-- 유전체 분석부 -->
    <g transform="translate(50, 500)">
      <rect width="700" height="80" rx="5" ry="5" fill="#e0f2f1" stroke="#009688" stroke-width="2"/>
      <text x="350" y="30" font-family="Arial" font-size="16" text-anchor="middle" font-weight="bold" fill="#004d40">
        유전체 분석부 (800)
      </text>
      
      <g transform="translate(150, 45)">
        <rect width="180" height="25" rx="5" ry="5" fill="#b2dfdb" stroke="#4db6ac" stroke-width="1"/>
        <text x="90" y="17" font-family="Arial" font-size="14" text-anchor="middle" fill="#333">
          유전자 시퀀싱 모듈
        </text>
      </g>
      
      <g transform="translate(370, 45)">
        <rect width="180" height="25" rx="5" ry="5" fill="#b2dfdb" stroke="#4db6ac" stroke-width="1"/>
        <text x="90" y="17" font-family="Arial" font-size="14" text-anchor="middle" fill="#333">
          위험도 예측 모듈
        </text>
      </g>
    </g>
    
    <!-- 데이터 흐름 화살표 -->
    <!-- 분석부 -> 자극 생성부 -->
    <path d="M400 180 L400 190" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 자극 생성부 -> 출력부 -->
    <path d="M160 290 L160 300" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M400 290 L400 300" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    <path d="M640 290 L640 300" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 수면 관리부 -> 출력부 -->
    <path d="M215 390 L215 380 L400 380 L400 380" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 적응형 제어부 -> 자극 생성부 -->
    <path d="M585 390 L585 380 L700 380 L700 240 L670 240" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
    
    <!-- 마커 정의 -->
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
      </marker>
    </defs>
    
    <!-- 애니메이션 요소: 데이터 흐름 표현 -->
    <circle r="5" fill="#2196f3">
      <animateMotion path="M400 130 L400 190 L160 190 L160 290 L160 340" dur="3s" repeatCount="indefinite"/>
    </circle>
    
    <circle r="5" fill="#4caf50">
      <animateMotion path="M400 130 L400 190 L400 190 L400 290 L400 340" dur="3s" repeatCount="indefinite" begin="0.5s"/>
    </circle>
    
    <circle r="5" fill="#03a9f4">
      <animateMotion path="M400 130 L400 190 L640 190 L640 290 L640 340" dur="3s" repeatCount="indefinite" begin="1s"/>
    </circle>
    
    <circle r="5" fill="#f44336">
      <animateMotion path="M215 390 L215 380 L400 380 L400 340" dur="2s" repeatCount="indefinite" begin="1.5s"/>
    </circle>
    
    <circle r="5" fill="#ffeb3b">
      <animateMotion path="M585 390 L585 380 L700 380 L700 240 L670 240" dur="4s" repeatCount="indefinite" begin="2s"/>
    </circle>
  </svg>
</div>

## 프로젝트 개요

이 저장소는 이명(Tinnitus) 환자를 위한 개인 맞춤형 복합 자극 기반 치료 시스템의 구현 코드와 연구 자료를 포함하고 있습니다. 이 시스템은 청각, 시각 및 촉각 자극을 복합적으로 제공하고 수면 상태를 개선하여 이명을 효과적으로 치료하기 위한 혁신적인 접근법을 제시합니다.

## 시스템 구성

이 시스템은 다음과 같은 주요 구성 요소로 이루어져 있습니다:

### 1. 다중 생체신호 분석부 (100)

- **고해상도 청력 검사 모듈 (110)**: 125Hz부터 16kHz까지 1/24 옥타브 간격으로 청력 역치를 측정하고, 이명 주파수를 0.1kHz 단위로 정밀하게 측정하여 3D 청각 지도를 작성
- **뇌파 분석 모듈 (120)**: 256채널 고밀도 EEG를 이용한 주파수 대역별 활성도 분석
- **수면 다원검사 모듈 (130)**: 수면의 질과 구조 분석을 위한 다원검사 데이터 처리

### 2. 청각 자극 생성부 (200)

- **노치 필터 모듈 (210)**: 환자의 이명 주파수 대역을 제거한 광대역 잡음 생성
- **주파수 변조 모듈 (220)**: 환자의 알파파 리듬과 동기화된 주파수 변조 적용
- **진폭 변조 모듈 (230)**: 델타파 리듬에 맞춘 진폭 변조 추가

### 3. 시각 자극 생성부 (300)

- **패턴 생성 모듈 (310)**: 청각 자극과 연동된 기하학적 패턴 생성
- **색상 최적화 모듈 (320)**: 환자 선호도와 심리 상태를 고려한 색상 조합 선택

### 4. 촉각 자극 생성부 (400)

- **진동 패턴 생성 모듈 (410)**: 청각 자극의 리듬과 동기화된 진동 패턴 생성
- **강도 조절 모듈 (420)**: 환자의 편안함과 치료 효과를 고려한 진동 강도 미세 조정

### 5. 복합 자극 출력부 (500)

- **청각 자극 출력 모듈 (510)**: 골전도 헤드폰을 통한 소리 전달
- **시각 자극 출력 모듈 (520)**: 마이크로 LED 내장 스마트 안대를 통한 시각 자극 제공
- **촉각 자극 출력 모듈 (530)**: 두피 부착형 진동 장치를 통한 진동 자극 전달

### 6. 수면 관리부 (600)

- **수면 단계 감지 모듈 (610)**: 실시간 뇌파 분석을 통한 수면 단계 판별
- **자극 최적화 모듈 (620)**: 각 수면 단계에 적합한 자극 패턴 선택 및 적용

### 7. 적응형 제어부 (700)

- **반응 분석 모듈 (710)**: 실시간 EEG 및 심박 변이도 데이터 분석
- **자극 최적화 모듈 (720)**: 강화학습 알고리즘을 이용한 개인별 최적 자극 패턴 업데이트

### 8. 유전체 분석부 (800)

- **유전자 시퀀싱 모듈 (810)**: 차세대 시퀀싱 기술을 이용한 환자의 전체 유전체 분석
- **위험도 예측 모듈 (820)**: 머신러닝 알고리즘을 이용한 유전적 요인 기반 이명 발생 위험도 계산

## 주요 특징

1. **개인 맞춤형 접근**: 고해상도 청력 검사, 뇌파 분석, 수면 패턴 평가를 통해 각 환자의 이명 특성을 정확히 파악
2. **복합 감각 자극**: 청각, 시각, 촉각 자극을 동시에 제공하여 뇌의 가소성을 극대화
3. **수면 연동 치료**: 수면 단계별 최적화된 자극 제공으로 수면 품질 개선과 이명 치료 동시 달성
4. **지속적 최적화**: 강화학습 기반 적응형 제어 시스템으로 개인별 치료 효과 최대화
5. **예방적 접근**: 유전체 분석을 통한 이명 발생 위험 예측 및 조기 대응

## 작동 원리

1. **분석 단계**: 환자의 청력, 뇌파, 수면 패턴을 종합적으로 분석하여 이명 특성을 파악하고 개인별 프로필 생성
2. **자극 생성 단계**: 분석 결과에 기반하여 청각, 시각, 촉각 자극을 생성하고 상호 동기화
3. **자극 출력 단계**: 웨어러블 디바이스(골전도 헤드폰, 스마트 안대, 진동 장치)를 통해 복합 자극 전달
4. **수면 관리 단계**: 실시간 수면 단계 감지를 통해 각 단계에 최적화된 자극 제공
5. **적응형 제어 단계**: 환자의 치료 반응을 지속적으로 모니터링하고 자극 패턴 최적화
6. **유전체 분석 단계**: 이명의 유전적 위험 요인 분석 및 예방 전략 수립

## 저장소 구조

- `Scientific_papers/`: 관련 연구 논문 및 자료
- `src/`: 소스 코드
  - `analysis/`: 다중 생체신호 분석 모듈
  - `stimuli/`: 청각, 시각, 촉각 자극 생성 모듈
  - `output/`: 복합 자극 출력 모듈
  - `sleep/`: 수면 관리 모듈
  - `adaptive/`: 적응형 제어 모듈
    - `action_space.py`: 자극 액션 공간 정의
    - `feature_extractor.py`: 환자 상태 특성 추출
    - `reward_function.py`: 강화학습 보상 함수
    - `state_tracker.py`: 환자 상태 및 반응 추적
    - `agent.py`: 강화학습 에이전트
    - `controller.py`: 적응형 제어 통합 컨트롤러
    - `config.py`: 강화학습 설정 파라미터
  - `genomics/`: 유전체 분석 모듈
  - `utils/`: 유틸리티 함수
- `examples/`: 예제 코드
  - `adaptive_controller_example.py`: 적응형 제어부 사용 예제
- `deployment/`: 배포 관련 문서 및 코드

## 사용 방법

### 설치 요구사항

```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 기본 사용법

```python
from src.analysis.audio_analysis import HighResolutionAudioAnalysis
from src.stimuli.auditory_stimuli import AuditoryStimGenerator

# 청력 분석 수행
analyzer = HighResolutionAudioAnalysis()
audiogram = analyzer.measure_full_audiogram(callback_function)
tinnitus_data = analyzer.analyze_tinnitus_frequency(audiogram, callback_function)

# 맞춤형 청각 자극 생성
generator = AuditoryStimGenerator()
brain_data = {'alpha_freq': 10.2, 'delta_freq': 1.8}
stimuli = generator.create_personalized_stimuli(
    tinnitus_freq=tinnitus_data['primary_frequency'],
    brain_data=brain_data,
    sound_type='pink_noise',
    sleep_stage='N2'
)

# 자극 저장
filepath = generator.save_stimuli(stimuli, 'therapy_session_001')
```

### 적응형 제어 시스템 사용법

```python
from src.adaptive import AdaptiveController

# 컨트롤러 초기화
controller = AdaptiveController()

# 환자 ID 설정 및 제어 시작
patient_id = "patient001"
controller.start(patient_id)

# 환자 상태 업데이트
controller.update_state(
    eeg_data=eeg_data,
    hrv_data=hrv_data,
    tinnitus_data=tinnitus_data,
    sleep_data=sleep_data
)

# 최적 자극 액션 얻기
action = controller.get_action(sleep_stage="N2")

# 자극 적용 후 환자 피드백 제공
feedback = {
    'tinnitus_intensity_change': -0.2,  # 이명 강도 감소
    'comfort': 0.8,                     # 환자 편안함 (0-1)
    'sleep_improvement': 0.6            # 수면 개선 정도 (0-1)
}
reward = controller.provide_feedback(feedback)

# 모델 학습
train_stats = controller.train()

# 제어 종료
controller.stop()
```

자세한 사용 예제는 `examples/adaptive_controller_example.py` 파일을 참조하세요.

## 적응형 강화학습 시스템

강화학습 기반 적응형 제어 시스템은 이명 치료의 핵심적인 혁신으로, 다음과 같은 특징을 가집니다:

1. **자극 액션 공간**: 청각, 시각, 촉각 자극의 다양한 파라미터 조합을 정의하여 가능한 모든 치료 자극 패턴을 표현
2. **상태 추적**: 환자의 EEG, HRV, 이명 특성, 수면 상태 등을 추적하고 이력 관리
3. **개인화된 보상 함수**: 이명 강도 감소, 수면 개선, 환자 편안함 등 다양한 지표를 종합적으로 평가하여 보상 산출
4. **복합 강화학습 알고리즘**: 
   - PyTorch 기반 DQN(Deep Q-Network) 구현(하드웨어 지원 시)
   - 간소화된 Q-테이블 기반 학습(기본 환경)
   - 실시간 탐색-활용 균형 조정
5. **수면 단계 연동**: 각 수면 단계(N1, N2, N3, REM)에 최적화된 자극 패턴 제공

이 시스템은 지속적인 환자 모니터링과 피드백을 통해 치료 효과를 지속적으로 개선하며, 개인별 맞춤형 치료를 가능하게 합니다.

## 향후 계획

1. 하드웨어 통합: 사용자 친화적인 웨어러블 디바이스 개발
2. 모바일 앱 개발: 치료 모니터링 및 제어를 위한 스마트폰 애플리케이션
3. 대규모 임상 시험: 치료 효과 검증 및 치료 프로토콜 최적화
4. 인공지능 강화: 딥러닝 모델을 통한 이명 패턴 분석 및 예측 개선
5. 원격 의료 통합: 의료진이 원격으로 환자의 치료 진행 상황을 모니터링하고 조정할 수 있는 시스템 개발

## 기여 방법

이 프로젝트에 기여하고 싶으시다면, 다음 과정을 따라주세요:

1. 이 저장소를 포크(Fork)하세요
2. 새로운 기능 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경 사항을 커밋하세요 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다.

## 연락처

개발자 연락처: [jjshome@example.com](mailto:jjshome@example.com)

## Patent Pending

본 기술은 특허 출원 중입니다.
