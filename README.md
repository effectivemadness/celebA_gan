# celebA_gan
* 단순 MLP를 사용한 GAN으로 celebA 얼굴 이미지를 학습, 생성한다.
* pytorch를 사용, celebA dataset은 resize해서 따로 생성해둔것을 loader로 로드한다.

* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) 참고하여 작성.
***
* Discriminator : FC-LeakyReLU-FC-LeakyReLU-FC-Sigmoid
* Generator : FC-BatchNorm1d-ReLU-FC-BatchNorm1d-ReLU-FC-Tanh

* Loss ftn : Mean Squared Error
* Optimizer : Adam
***
* Discriminator : random Z값으로 fake, Real data 분별하게 학습
* Generator : random Z값으로 생성한 fake를 discriminator가 real로 판별하도록 학습

* 생성/분별 시 RGB를 따로 처리할 필요가 있음. 따라서 (image size)*3의 학습 용량을 가짐.

* 학습과정을 보기 위해 random Z값이 아닌 고정된 Z값으로 생성한 이미지도 출력.

| epoch | result | result |
|:---:|:---:|:---:|:---:|
| 0 | ![epoch0](./images/epoch0.png) | ![epoch0](./images/epoch0_fix.png) |
| 100 | ![epoch100](./images/epoch50.png) | ![epoch0](./images/epoch50_fix.png) |
| 100 | ![epoch100](./images/epoch100.png) | ![epoch0](./images/epoch100_fix.png) |
| 200 | ![epoch200](./images/epoch200.png) | ![epoch0](./images/epoch200_fix.png) |