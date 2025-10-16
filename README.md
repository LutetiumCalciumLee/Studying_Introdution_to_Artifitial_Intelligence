<details>
<summary>ENG (English Version)</summary>

# Introduction to Artificial Intelligence

***

### 1. Introduction to Artificial Intelligence
- **AI Levels and Applications:** AI varies in complexity and is used in technologies like object detection (e.g., YOL-NAS) and in consumer products like smart vacuum cleaners.
- **Service Providers:** Companies like Naver (Clova) and Kakao (Kakao i) offer AI-driven services.
- **AI+x:** AI is being integrated into various fields.
- **Intelligence:** Defined by key abilities such as reasoning, learning, and problem-solving. Natural intelligence is the benchmark.
- **AI Definition and History:** AI is the simulation of human intelligence in machines. Its history is marked by a timeline of key developments and different approaches.
- **AI Types & Future:** AI can be categorized (e.g., narrow vs. general). Its development is predicted to bring significant societal changes.
- **Key Technologies:** Core technologies drive AI's capabilities.
- **Judging Machine Intelligence:** Evaluating machine intelligence is a complex challenge. The goal is not always to perfectly imitate human intelligence.
- **Pros and Cons:** AI offers significant advantages but also comes with potential drawbacks. Understanding AI is crucial in the modern world.

***

### 2. Deep Learning Environment
- **Setup Considerations:** Setting up a deep learning environment requires careful thought about programming languages (like Python) and frameworks (like TensorFlow or PyTorch).

***

### 3. AI, Machine Learning (ML), and Deep Learning (DL)
- **Relationships:** DL is a subfield of ML, which is a subfield of AI.
- **ML vs. DL:** Deep learning uses deep neural networks and often handles unstructured data better than traditional machine learning.
- **ML Frameworks & Categories:** Various frameworks support ML algorithms, which are categorized into supervised, unsupervised, semi-supervised, and reinforcement learning.
- **Supervised Learning:** Learns from labeled data to make predictions. Applications include classification and regression.
- **Unsupervised Learning:** Finds patterns in unlabeled data. Applications include clustering and dimensionality reduction.
- **Semi-supervised Learning:** Uses a mix of labeled and unlabeled data.
- **Reinforcement Learning:** Learns by trial and error through rewards and penalties.

***

### 4. Deep Learning (DL) Basics
- **Inspiration:** DL is inspired by the human nervous system.
- **Perceptron:** The simplest form of a neural network, imitating a single neuron.
- **Artificial Neural Networks (ANN):** Networks of interconnected perceptrons.
- **Deep Neural Networks (DNN):** ANNs with multiple hidden layers between the input and output layers.
- **DL Training:** The process involves feeding data to the network and adjusting its parameters to minimize errors, which is known as training.

***

### 5. DL & ANN In-Depth
- **Why DL Matters:** Deep learning is significant now due to increased data availability, powerful computing hardware, and advanced algorithms.
- **Neural Network Processing:** Information flows through the network in a process called forward propagation.
- **Loss Measurement:** The network's error is quantified by a loss function (e.g., Binary Cross Entropy, Mean Squared Error).
- **Network Training:** The goal is to minimize this loss through optimization algorithms. This process can be challenging due to complex, high-dimensional data.

***

### 6. Convolutional Neural Networks (CNN)
- **Computer Vision:** A field of AI that enables computers to "see" and interpret visual information from the world. It has wide-ranging applications and impacts.
- **Image Data:** Computers perceive images as numerical data (pixels).
- **CNN for Vision:** Traditional neural networks struggle with the high dimensionality and spatial information of images. CNNs are designed to address this.
- **Convolution Operation:** CNNs use convolution to extract features (like edges, textures) from images by sliding a filter over them, creating a feature map.
- **CNN Architecture:** Consists of convolutional layers that learn a hierarchy of features (representation learning).
- **Applications:** CNNs are used for image classification, object detection (e.g., YOLO), semantic segmentation, and control systems.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 인공지능 개론

***

### 1. 인공지능 소개
- **AI 수준 및 활용:** AI는 다양한 수준으로 존재하며, 객체 탐지(YOL-NAS 등)나 AI 청소기 같은 제품에 활용됨.
- **서비스 제공사:** 네이버(클로바), 카카오(Kakao i) 등이 AI 서비스를 제공.
- **AI+x:** AI가 다양한 분야와 융합되는 현상.
- **지능의 정의:** 지능은 추론, 학습, 문제 해결 등의 능력으로 정의되며, 자연 지능이 그 기준이 됨.
- **AI 정의와 역사:** AI는 인간 지능을 기계로 모방하는 것으로, 주요 사건과 접근법의 변화로 역사가 구성됨.
- **AI 유형 및 미래:** AI는 기능에 따라 분류(약인공지능, 강인공지능 등)되며, 발전은 사회에 큰 변화를 가져올 것으로 예측됨.
- **핵심 기술:** AI의 발전을 이끄는 핵심 기술들이 존재함.
- **기계 지능 판단의 어려움:** 기계의 지능을 판단하는 것은 복잡한 문제이며, AI의 목표가 반드시 인간 지능의 완벽한 모방은 아님.
- **장단점:** AI는 큰 장점을 가지지만 단점도 존재함. AI 학습은 현대 사회에서 중요함.

***

### 2. 딥러닝 환경
- **환경 구축 고려사항:** 딥러닝 환경 구축 시 프로그래밍 언어(주로 파이썬)와 프레임워크(텐서플로우, 파이토치 등)를 신중히 선택해야 함.

***

### 3. AI, 머신러닝(ML), 딥러닝(DL)
- **관계:** DL은 ML의 한 분야이며, ML은 AI의 한 분야임.
- **ML vs DL:** 딥러닝은 심층 신경망을 사용해 비정형 데이터 처리에 강점을 보임.
- **ML 프레임워크 및 분류:** 다양한 프레임워크가 존재하며, 학습 방식에 따라 지도, 비지도, 준지도, 강화학습으로 나뉨.
- **지도학습:** 정답이 있는 데이터로 학습하여 예측. 분류, 회귀 문제에 적용.
- **비지도학습:** 정답 없는 데이터에서 패턴 발견. 군집화, 차원 축소 등에 사용.
- **준지도학습:** 일부 정답 데이터와 다수 비정답 데이터를 함께 사용.
- **강화학습:** 보상과 벌점을 통해 시행착오를 거치며 학습.

***

### 4. 딥러닝(DL) 기초
- **영감:** 딥러닝은 인간의 신경계에서 영감을 받음.
- **퍼셉트론:** 뉴런 하나를 모방한 가장 간단한 형태의 신경망.
- **인공 신경망 (ANN):** 여러 퍼셉트론이 연결된 네트워크.
- **심층 신경망 (DNN):** 입력층과 출력층 사이에 여러 개의 은닉층을 가진 인공 신경망.
- **DL 학습:** 데이터를 신경망에 입력하고, 오차를 줄이도록 매개변수를 조정하는 과정.

***

### 5. 딥러닝과 인공 신경망 심화
- **딥러닝의 의의:** 풍부한 데이터, 강력한 하드웨어, 발전된 알고리즘 덕분에 현재 주목받고 있음.
- **신경망 정보 처리:** 정보는 순전파(Forward Propagation) 과정을 통해 신경망을 통과함.
- **손실 측정:** 신경망의 오차는 손실 함수(이진 교차 엔트로피, 평균 제곱 오차 등)로 정량화됨.
- **신경망 학습:** 최적화 알고리즘을 통해 손실을 최소화하는 과정으로, 복잡하고 고차원적인 데이터 때문에 어려움이 따름.

***

### 6. 합성곱 신경망 (CNN)
- **컴퓨터 비전:** 컴퓨터가 시각 정보를 "보고" 해석하게 하는 AI 분야로, 응용 분야와 영향력이 매우 넓음.
- **이미지 데이터:** 컴퓨터는 이미지를 픽셀 값의 집합(숫자)으로 인식함.
- **비전 문제를 위한 CNN:** 전통적인 신경망은 이미지의 고차원성과 공간적 정보 처리에 한계가 있어 CNN이 개발됨.
- **합성곱 연산:** CNN은 필터를 이미지 위에서 이동시키며 합성곱 연산을 통해 특징(모서리, 질감 등)을 추출하고, 이를 특징 맵(Feature Map)으로 만듦.
- **CNN 구조:** 합성곱 계층을 통해 특징의 계층 구조를 학습(표현 학습)함.
- **응용:** 이미지 분류, 객체 탐지(YOLO 등), 시맨틱 분할, 제어 시스템 등 다양한 분야에 활용됨.

</details>
