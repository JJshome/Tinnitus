# 개인 맞춤형 복합 자극 기반 이명 치료 시스템 설치 가이드

이 가이드는 개인 맞춤형 복합 자극 기반 이명 치료 시스템의 설치 및 설정 방법을 안내합니다.

## 시스템 요구사항

### 소프트웨어 요구사항

- Python 3.8 이상
- Anaconda 또는 Miniconda (권장)
- Git

### 하드웨어 요구사항

#### 개발/분석 환경
- CPU: 최소 4코어 (8코어 이상 권장)
- RAM: 최소 16GB (32GB 이상 권장)
- GPU: NVIDIA GeForce RTX 3060 이상 (딥러닝 기반 분석 시)
- 저장 공간: 최소 100GB (유전체 분석 시 1TB 이상 권장)

#### 치료 시스템 하드웨어
- 골전도 헤드폰
- 마이크로 LED 내장 스마트 안대
- 두피 부착형 진동 장치
- EEG 모니터링 장비 (256채널 고밀도 EEG)
- 블루투스 5.0 이상 지원 데이터 전송 모듈
- 배터리 모듈 (최소 12시간 작동 가능)

## 설치 과정

### 1. 저장소 복제

```bash
git clone https://github.com/JJshome/Tinnitus.git
cd Tinnitus
```

### 2. 가상 환경 설정

```bash
# Anaconda 사용 시
conda create -n tinnitus python=3.8
conda activate tinnitus

# venv 사용 시
python -m venv tinnitus_env
source tinnitus_env/bin/activate  # Linux/macOS
# 또는
tinnitus_env\Scripts\activate  # Windows
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt` 파일 내용:

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
soundfile>=0.10.3
scikit-learn>=1.0.0
pytorch>=1.10.0
torchvision>=0.11.0
mne>=0.23.0
pyedflib>=0.1.22
pyserial>=3.5
joblib>=1.1.0
tqdm>=4.62.0
h5py>=3.6.0
pyyaml>=6.0
PyQt5>=5.15.0  # GUI 개발용
pygatt>=4.0.5  # 블루투스 통신용
```

### 4. 하드웨어 설정

#### 4.1 골전도 헤드폰 설정

1. 블루투스 페어링 설정
   ```bash
   python src/setup/bluetooth_pairing.py --device headphone
   ```

2. 오디오 출력 설정
   ```bash
   python src/setup/audio_setup.py --device bone_conduction
   ```

#### 4.2 스마트 안대 설정

1. 마이크로 LED 테스트
   ```bash
   python src/setup/led_test.py
   ```

2. 블루투스 페어링 설정
   ```bash
   python src/setup/bluetooth_pairing.py --device smart_eyemask
   ```

3. 색상 및 밝기 보정
   ```bash
   python src/setup/calibrate_led.py
   ```

#### 4.3 두피 부착형 진동 장치 설정

1. 블루투스 페어링 설정
   ```bash
   python src/setup/bluetooth_pairing.py --device vibration_device
   ```

2. 진동 모터 테스트
   ```bash
   python src/setup/vibration_test.py --intensity 50 --frequency 10
   ```

3. 부착 위치 최적화 테스트
   ```bash
   python src/setup/position_optimization.py
   ```

#### 4.4 EEG 장비 설정

1. 드라이버 설치
   - 제조사 웹사이트에서 최신 EEG 하드웨어 드라이버 다운로드 및 설치

2. EEG 연결 테스트
   ```bash
   python src/setup/eeg_connection_test.py
   ```

3. 채널 임피던스 테스트
   ```bash
   python src/setup/eeg_impedance_test.py
   ```

### 5. 시스템 초기 설정

#### 5.1 시스템 구성 파일 생성

1. 기본 구성 파일 생성
   ```bash
   python src/setup/generate_config.py
   ```

2. 구성 파일 편집 (하드웨어 및 환경에 맞게 수정)
   ```bash
   # config.yaml 파일을 직접 편집하거나 다음 명령어 사용
   python src/setup/configure_system.py --interactive
   ```

#### 5.2 데이터베이스 초기화

```bash
python src/setup/initialize_database.py
```

#### 5.3 테스트 실행

1. 시스템 통합 테스트
   ```bash
   python src/test/integration_test.py
   ```

2. 하드웨어 동기화 테스트
   ```bash
   python src/test/synchronization_test.py
   ```

## 사용자 인터페이스 설정

### 1. GUI 실행

```bash
python src/ui/main.py
```

### 2. 웹 인터페이스 설정 (선택 사항)

1. 웹 서버 실행
   ```bash
   python src/web/server.py
   ```

2. 웹 인터페이스 접속
   - 웹 브라우저를 열고 `http://localhost:8080` 접속

## 문제 해결

### 하드웨어 연결 문제

1. 블루투스 연결 문제
   ```bash
   python src/troubleshoot/bluetooth_diagnosis.py
   ```

2. 오디오 장치 문제
   ```bash
   python src/troubleshoot/audio_diagnosis.py
   ```

3. EEG 신호 문제
   ```bash
   python src/troubleshoot/eeg_diagnosis.py
   ```

### 소프트웨어 문제

1. 로그 확인
   ```bash
   cat logs/system.log
   ```

2. 진단 테스트 실행
   ```bash
   python src/troubleshoot/system_diagnosis.py
   ```

3. 구성 재설정
   ```bash
   python src/setup/reset_config.py
   ```

## 업데이트 방법

시스템을 최신 버전으로 업데이트하려면:

```bash
git pull origin main
pip install -r requirements.txt
python src/setup/update_system.py
```

## 백업 및 복원

### 1. 시스템 백업

```bash
python src/utils/backup.py --output backup_YYYY-MM-DD
```

### 2. 백업에서 복원

```bash
python src/utils/restore.py --input backup_YYYY-MM-DD
```

## 임상 데이터 관리

### 1. 환자 데이터 내보내기

```bash
python src/utils/export_data.py --patient_id <ID> --format csv
```

### 2. 익명화된 집계 데이터 생성

```bash
python src/utils/anonymize_data.py --output anonymized_data.csv
```

### 3. 데이터 분석 보고서 생성

```bash
python src/analysis/generate_report.py --patient_id <ID> --output report.pdf
```

## 시스템 사양 맞춤화

특정 하드웨어와 병원 환경에 맞게 시스템을 조정하려면:

```bash
python src/setup/customize_system.py --environment hospital --devices "eeg_model_x,headphone_model_y"
```

## 성능 모니터링

시스템 성능 모니터링을 위한 대시보드 실행:

```bash
python src/utils/dashboard.py
```

브라우저에서 `http://localhost:8050`으로 접속하여 실시간 성능 데이터를 확인할 수 있습니다.

---

이 가이드에 대한 질문이나 문제 해결을 위해 [jjshome@example.com](mailto:jjshome@example.com)으로 문의하거나 GitHub 이슈를 생성해 주세요.
