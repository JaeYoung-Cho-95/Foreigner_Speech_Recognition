# Foreigner_Speech_Recognition
외국인 발화 한국어 인식 공모전.

" Cslee와 NIA에서 주관하는 외국인이 발화하는 한국어 데이터를 활용한 인공지능 공모전입니다. "
- 평가항목은 인공지능 모델의 성능, 사업화 아이디어 2가지 였습니다.
- 대상이 아니여서 아쉬웠지만, 우수상을 받았으며 해당 과정에 대해 정리하겠습니다.
- 문제점, 목차, kospeech deepspeech2 사용법 소개하는 식으로 진행하겠습니다.
- 문제점에 대해 먼저 말씀드리는 순서는 이후에 소개할 목차가 다소 이해하기 난해한 순서로 진행되었기 때문입니다.

## 문제점
- 인공지능 모델 개발, 사업화 아이디어 제시를 3주라는 짧은 시간 안에 제시해야 했다.
- 데이터를 공모전 시작 1주일 후에 제공 받았다.
- 처음에 주최측에서 제시하는 Kaldi 라는 STT 모델이 있었는데 권장사양이 64기가 였고, 이 정도의 로컬환경을 가진 팀원이 없었다.
- Colab Pro Plus 를 제공해주기로 하여 로컬환경이 없는 친구들과 함께 기다리고 있었는데 공모전 시작 2주일 후에 제공받았다.
- 주최측에서 제공하는 Colab을 이용하기 위해서는 데이터를 Google Drive 에 올려서 사용해야했는데 zip파일을 포함해 200GB가 넘는 용량이였기 때문에 클라우드 공간에 올리는데만 3일이 걸렸다.


## 사업화 아이디어 및 논문 탐색
### (1) 사업화 아이디어 회의
- 데이터가 없지만 Orientation 파일을 통해 먼저 사업화 아이디어 회의를 진행했습니다.
- 외국인들의 한국어 발화 데이터를 통해 만들 수 있다고 판단된 아이디어는 "한국어 교육 애플리케이션" , "한국어 공부를 원하는 외국인들을 위한 발음 교정 어플리케이션", "음성인식 인공지능을 활용한 한국어 능력시험" 이 있었습니다.
- 어플리케이션 개발은 이미 존재하는 앱들이 많았고 실제 사용하는 외국인이 많지 않다는 점을 감안해 "음성인식 인공지능을 활용한 한국어 능력시험"을 일차적으로 채택하게 되었습니다.
### (2) 논문 탐색
- STT 모델의 종류, 성능, 원리에 대한 논문 탐색이 이루어졌습니다.
- 위 내용을 제외하고는 한국인이 말한 데이터를 Pre-Trained 한 Model에 fine-tuning 을 진행할 것인가, 외국인 발화에 대한 데이터만으로 학습을 진행하는 것이 더 좋은 성능을 낼 것인가에 대한 탐색이 가장 큰 비중을 차지했습니다.
## 3. 한국어 음성 데이터를 이용한 적절한 STT 모델 찾아보기
- 좋은 성능을 내는 STT모델은 많이 있었습니다. 하지만 한국어에 대해 성능 지표가 제재로 나와있는 모델은 많지 않았습니다.
- 시간이 없는 이유로 한 번이라도 한국어로 선행된 모델이을 먼저 찾았고, 그 중에 성능을 비교해 우수하다고 판단되는 모델을 선택하게 되었습니다.
## 4. 간단한 Refactoring 및 훈련
- https://github.com/sooftware/kospeech 모델 소스 입니다.(개발에 힘써주신 김 수환 님 감사드립니다!)
- kospeech 속에 있는 DeepSpeech2 모델을 사용하기로 결정했습니다.
- Refactoring 내용은 아래에 정리되어있습니다.
## 5. 사업화 아이디어 정리 및 PPT 제작

## Kospeech DeepSpeech2 train사용법
- 공모전은 Colab 을 이용해 돌렸지만 공모전 종류 후 Local 환경에서 돌려보기 위해 다양한 시도를 했으며, 그 과정 속에서 발생한 오류를 해결하기 위한 방법들을 정리하는 내용입니다.
### (1) kospeech에서 제공해주는 setup.py파일을 통해 pip install -e . 를 실행하면 Levenshtein install error 가 발생한다.
 - pip install python-Levenshtein은 오류가 발생
 - conda install -c conda-forge python-levenshtein 를 하면 오류 해결
### (2) kospeech의 setup파일에 있는 pytorch 버전은 나의 로컬 환경인 rtx 3070ti, cuda 11.2 버전과 호환이 되지않아 pytorch만 cuda호환이 되는 버전으로 재설치 진행
- 나의 환경과 같은 경우 아래 코드로 해결
- pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 
### (3) python ./bin/main.py model=ds2 train=ds2_train train.dataset_path=$DATASET_PATH 명령 실행 
- main.py 파일에 들어와봤는 import hydra 설치가 진행되지 않았음
  - pip install hydra-core 명령어 실행을 통해 해결
  
- ModuleNotFoundError: No module named 'kospeech'
   - 모듈 불러오는 경로가 잘못잡혀있음
   - kospeech 폴더 자체를 bin안으로 넣어줌
   
- ModuleNotFoundError: No module named 'librosa'
   - pip install librosa 로 해결
   
- ModuleNotFoundError: No module named 'astropy'
   - pip install astropy
   
- from kospeech.models.las.decoder import DecoderRNN, BeamDecoderRNN
ImportError: cannot import name 'BeamDecoderRNN' from 'kospeech.models.las.decoder' (C:\Users\JaeYoungCho\Desktop\한국어외국인발화음성인식\code\kospeech_latest\bin\kospeech\models\las\decoder.py)
   - 'BeamDecoderRNN' import 추석처리
   
- ModuleNotFoundError: No module named 'pandas'
  - pip install pandas
  
- hydra grammer 오류
  - 경로에 한글 있는 것 영어로 바꿔줌 
  - 한국어외국인발화음성인식 > Foreigner_Speech_Recognition
  
- FileNotFoundError: [Errno 2] No such file or directory: '../../../data/transcripts.txt'
  - 음성파일의 label 값들이 포함되어 있는 json 에서 뽑아와야 할 듯
  - 파일경로/파일이름 라벨값 으로 정리되어있는 txt 로 넣어줌
  
- UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc0 in position 30: invalid start byte
  - text 파일을 utf-8로 다시 인코딩해줬다

- File "C:\Users\JaeYoungCho\Desktop\Foreigner_Speech_Recognition\code\kospeech_latest\bin\kospeech\data\data_loader.py", line 106, in shuffle
    self.audio_paths, self.transcripts, self.augment_methods = zip(*tmp)
ValueError: not enough values to unpack (expected 3, got 0)
  - data_loader 중간에 있는 split data의 길이를 우리가 가지고 있는 30만개의 데이터를 기준으로 10%만 검증용으로 주고 나머지는 train으로 맞춰줬다.

- RuntimeError: CUDA out of memory. Tried to allocate 792.00 MiB (GPU 0; 8.00 GiB total capacity; 5.78 GiB already allocated; 70.00 MiB free; 5.95 GiB reserved in total by PyTorch)
  - ds2_train.yaml 파일찾아서 batch_size 줄여주기
  - import gc   ,   gc.collect() ,   torch.cuda.empty_cache() 명령어 실행

- Audio is None Error 가 발생한다. 
  - 경로 설정 제대로 해주기

> Train 성공!



 
