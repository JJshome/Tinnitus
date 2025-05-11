# 개인 맞춤형 복합 자극 기반 이명 치료 시스템 배포 가이드

이 폴더에는 개인 맞춤형 복합 자극 기반 이명 치료 시스템의 배포와 관련된 코드 및 문서가 포함되어 있습니다.

## 목차

1. [시스템 요구사항](#1-시스템-요구사항)
2. [설치 가이드](#2-설치-가이드)
3. [환경 설정](#3-환경-설정)
4. [사용자 인터페이스 설정](#4-사용자-인터페이스-설정)
5. [하드웨어 연결 가이드](#5-하드웨어-연결-가이드)
6. [테스트 프로토콜](#6-테스트-프로토콜)
7. [문제 해결](#7-문제-해결)

## 1. 시스템 요구사항

시스템 요구사항에 대한 상세 정보는 [system_requirements.md](system_requirements.md) 문서를 참조하세요.

### 주요 요구사항 요약

- **OS**: Windows 10/11, macOS 11.0+, Ubuntu 20.04 LTS+
- **프로세서**: Intel Core i5 8세대+ 또는 AMD Ryzen 5 3600+
- **메모리**: 최소 8GB RAM (권장 16GB)
- **저장공간**: 최소 20GB 여유 공간 (SSD 권장)
- **그래픽**: 딥러닝 모델을 위한 NVIDIA GTX 1650+ (선택사항)
- **연결성**: Bluetooth 5.0, USB 3.0 포트

## 2. 설치 가이드

### 자동 설치

```bash
# 설치 스크립트 실행
python install.py

# 옵션
python install.py --skip-deps        # 의존성 설치 건너뛰기
python install.py --skip-hw-check    # 하드웨어 확인 건너뛰기
```

### 수동 설치

1. Python 3.9 이상 설치
2. 필요한 패키지 설치: `pip install -r requirements.txt`
3. 환경 설정 파일 생성: `python -m deployment.scripts.create_config`
4. 하드웨어 테스트: `python -m deployment.scripts.hardware_test --all`

## 3. 환경 설정

시스템 구성은 `config/system_config.json` 파일을 통해 관리됩니다.

### 주요 설정 항목

```json
{
  "audio": {
    "sample_rate": 44100,
    "channels": 2,
    "device_id": "default"
  },
  "eeg": {
    "device_type": "auto",
    "channels": 32,
    "sample_rate": 256
  },
  "system": {
    "data_dir": "/path/to/data",
    "log_level": "INFO",
    "use_gpu": true
  }
}
```

### 설정 편집 도구

```bash
# GUI 설정 도구 실행 (추후 개발 예정)
python -m deployment.scripts.config_editor

# 특정 설정 업데이트
python -m deployment.scripts.update_config --set system.log_level=DEBUG
```

## 4. 사용자 인터페이스 설정

### 테마 설정

시스템은 다크 테마와 라이트 테마를 지원합니다. 테마는 다음과 같이 설정할 수 있습니다:

```bash
python -m deployment.scripts.update_config --set user_interface.theme=dark
```

### 언어 설정

현재 지원하는 언어: 한국어(ko), 영어(en), 일본어(ja)

```bash
python -m deployment.scripts.update_config --set user_interface.language=ko
```

### 접근성 설정

```bash
# 고대비 모드 활성화
python -m deployment.scripts.update_config --set user_interface.high_contrast=true

# 텍스트 크기 조정
python -m deployment.scripts.update_config --set user_interface.text_scale=1.5
```

## 5. 하드웨어 연결 가이드

하드웨어 설정에 대한 상세 정보는 [hardware_setup.md](hardware_setup.md) 문서를 참조하세요.

### 빠른 연결 가이드

1. EEG 장비 연결:
   - 장비 전원 켜기
   - 필요한 드라이버 설치
   - 시스템에 연결 후 `hardware_test.py --eeg` 실행하여 확인

2. 골전도 헤드폰 연결:
   - Bluetooth 페어링 모드 진입
   - OS에서 장치 연결
   - 기본 오디오 출력 장치로 설정

3. 시각/촉각 자극 장치 연결:
   - USB 케이블로 시스템에 연결
   - 장치 인식 확인
   - 펌웨어 업데이트 필요시 실행

## 6. 테스트 프로토콜

### 기능 테스트

```bash
# 전체 하드웨어 테스트
python -m deployment.scripts.hardware_test --all

# 특정 구성 요소 테스트
python -m deployment.scripts.hardware_test --eeg --audio
```

### 성능 테스트

```bash
# 자극 동기화 테스트
python -m deployment.scripts.sync_test

# 지연 시간 측정
python -m deployment.scripts.latency_test
```

### 배포 전 체크리스트

- [ ] 모든 하드웨어 구성 요소 연결 및 테스트 완료
- [ ] 시스템 설정 검토 및 사용자 환경에 맞게 조정
- [ ] 데이터 저장 디렉토리 권한 확인
- [ ] 백업 시스템 설정
- [ ] 사용자 계정 및 권한 설정
- [ ] 네트워크 구성 (필요시)

## 7. 문제 해결

### 일반적인 문제

| 문제 | 해결 방법 |
|------|---------|
| 하드웨어 인식 안됨 | 드라이버 재설치, 다른 USB 포트 시도, 장치 재부팅 |
| 소리가 들리지 않음 | 볼륨 설정 확인, 기본 장치 확인, 헤드폰 연결 상태 점검 |
| EEG 신호 품질 낮음 | 전극 위치 확인, 전해질 용액 추가, 주변 전자기 간섭 제거 |
| 시스템 응답 느림 | 리소스 모니터링, 불필요한 프로세스 종료, 메모리 확인 |

### 로그 분석

시스템 로그는 `data/logs` 디렉토리에 저장됩니다. 문제 해결을 위해 다음 로그 파일을 확인하세요:

- `system.log`: 일반 시스템 로그
- `hardware.log`: 하드웨어 관련 로그
- `stimulus.log`: 자극 생성 및 동기화 관련 로그
- `eeg.log`: EEG 데이터 처리 관련 로그

### 지원 요청

문제가 지속되는 경우 GitHub 이슈를 생성하여 도움을 요청하세요. 이슈 생성 시 다음 정보를 포함해 주세요:

1. 시스템 환경 (OS, Python 버전, 하드웨어 구성)
2. 문제 상세 설명
3. 재현 단계
4. 로그 파일 첨부
