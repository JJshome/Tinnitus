# 하드웨어 설정 가이드

이 문서는 개인 맞춤형 복합 자극 기반 이명 치료 시스템의 하드웨어 구성 및 연결 방법을 설명합니다.

## 하드웨어 구성 요소

### 1. EEG 측정 장비 설정

#### 지원 장비
- Emotiv EPOC X / EPOC Flex
- OpenBCI Ultracortex Mark IV
- g.tec g.Nautilus
- Neuroelectrics Enobio

#### 연결 및 설정 방법
1. **드라이버 설치**:
   - 각 제조사 웹사이트에서 최신 드라이버 설치
   - Windows의 경우 장치 관리자에서 드라이버 상태 확인

2. **EEG 장비 준비**:
   ```
   - 전극이 건조하지 않도록 전해질 용액 적용
   - 헤드셋 착용 전 전극 위치 확인
   - 임피던스 점검: 모든 채널이 녹색(< 10kΩ)인지 확인
   ```

3. **소프트웨어 연결 확인**:
   ```bash
   # OpenBCI 예시
   python -c "import brainflow; board = brainflow.BoardShim(1, brainflow.BrainFlowInputParams()); print('연결 성공')"
   ```

### 2. 청각 자극 출력 설정

#### 지원 장비
- Shokz OpenRun Pro (골전도 헤드폰)
- Sennheiser HD 660 S (주변 소음 차단용)

#### 연결 및 설정 방법
1. **Bluetooth 연결** (골전도 헤드폰):
   - 페어링 모드 진입: 전원 버튼 5초간 길게 누름
   - 시스템 Bluetooth 설정에서 장치 추가
   - 연결 후 기본 오디오 출력 장치로 설정

2. **유선 헤드폰**:
   - 오디오 인터페이스에 헤드폰 연결
   - 볼륨 레벨 설정: 중간 레벨(50%)에서 시작

3. **구성 파일 설정**:
   ```json
   {
     "audio": {
       "device_id": "Shokz OpenRun Pro",
       "sample_rate": 44100,
       "channels": 2,
       "buffer_size": 1024
     }
   }
   ```

### 3. 시각 자극 출력 설정

#### 지원 장비
- 커스텀 제작 LED 안대
- Philips Hue 연동 장치

#### 연결 및 설정 방법
1. **LED 안대 준비**:
   - 마이크로 컨트롤러(Arduino Nano/ESP32) 연결
   - USB 케이블로 시스템에 연결
   - 인식된 COM 포트 번호 확인

2. **펌웨어 업데이트**:
   ```bash
   # 펌웨어 업데이트 명령 (Arduino)
   arduino-cli upload -p COM4 --fqbn arduino:avr:nano deployment/firmware/visualstim.hex
   ```

3. **구성 파일 설정**:
   ```json
   {
     "visual": {
       "device_port": "COM4",  // Windows 예시
       "baudrate": 115200,
       "max_brightness": 80,
       "patterns": ["calm", "transient", "subtle"]
     }
   }
   ```

### 4. 촉각 자극 출력 설정

#### 지원 장비
- 커스텀 압전 진동 모듈
- 촉각 피드백 컨트롤러

#### 연결 및 설정 방법
1. **모듈 연결**:
   - 진동 모듈을 컨트롤러 보드에 연결
   - 컨트롤러를 USB 포트에 연결
   - 전원 스위치 ON 상태 확인

2. **드라이버 설치**:
   ```bash
   # Linux 예시
   sudo apt-get install libusb-1.0-0-dev
   # Windows: 제공된 드라이버 설치 프로그램 실행
   ```

3. **연결 테스트**:
   ```bash
   # 진동 모듈 테스트
   python deployment/scripts/test_haptic.py
   ```

4. **구성 파일 설정**:
   ```json
   {
     "haptic": {
       "device_port": "/dev/ttyUSB0",  // Linux 예시
       "intensity_levels": 10,
       "max_frequency": 200,
       "response_time": 20
     }
   }
   ```

## 통합 하드웨어 테스트

모든 장치가 올바르게 연결되었는지 확인하려면 통합 테스트를 실행하세요:

```bash
python deployment/scripts/hardware_test.py --all
```

또는 특정 구성 요소만 테스트:

```bash
python deployment/scripts/hardware_test.py --eeg --audio
```

## 문제 해결

### EEG 연결 문제
- **신호가 나타나지 않음**: 전극 접촉 상태 확인, 전해질 용액 다시 적용
- **노이즈가 심함**: 전자기 간섭 가능성 확인, 무선 장치 및 전원 어댑터와 거리 확보

### 오디오 문제
- **소리가 들리지 않음**: 볼륨 레벨 확인, 기본 장치 설정 확인
- **지연 발생**: 오디오 버퍼 크기 조정 (config.json에서 buffer_size 증가)

### 시각/촉각 자극 문제
- **장치가 응답하지 않음**: COM 포트 확인, 펌웨어 재설치
- **일관되지 않은 자극**: 전원 공급 확인, USB 허브 사용 시 직접 연결로 변경

## 하드웨어 성능 최적화

### EEG 신호 품질 개선
- 조용하고 전자기 간섭이 적은 환경에서 사용
- 측정 전 두피 준비: 알코올 면봉으로 약간 닦은 후 전극 부착
- 헤드셋 착용 15분 후 임피던스 재확인

### 자극 동기화 최적화
- 모든 자극 장치는 USB 3.0 이상 포트에 직접 연결
- CPU 부하가 높은 프로그램 종료
- 실시간 처리 우선순위 설정 활성화 (config.json에서 "priority": "high" 설정)

## 하드웨어 유지 관리

### 정기 점검 사항
- 전극 상태 확인 및 필요 시 교체 (대략 100시간 사용 후)
- 케이블 연결 상태 확인
- 펌웨어 월 1회 업데이트

### 청소 및 보관
- EEG 전극: 사용 후 증류수로 세척하고 건조
- 헤드폰: 알코올 솜으로 부드럽게 닦기
- 모든 장비는 습기가 없는 실온에 보관
