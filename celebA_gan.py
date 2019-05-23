
import torch #~pytorch 임포트
import torch.nn as nn #~pytorch 임포트
import torch.utils as utils #~pytorch 임포트
import torch.nn.init as init #~pytorch 임포트
from torch.autograd import Variable #~pytorch 임포트
import torchvision.utils as v_utils #~torchvision 임포트
import torchvision.datasets as dset #~torchvision 임포트
import torchvision.transforms as transforms #~torchvision 임포트
import numpy as np #numpy 임포트
import matplotlib.pyplot as plt #matplotlib임포트
from collections import OrderedDict #ordereddict 임포트
import time #time 임포트

epoch = 500 #학습 반복 epoch = 500
batch_size = 64 #한번에 돌아갈 batch size = 64
learning_rate = 0.0002 #학습률 = 0.0002
num_gpus = 1 #num_gpu = 1 gpu수 = 1 - 사용안함
z_size = 100 #생성모델 인풋으로 사용할 z size = 100
middle_size = 200 #MLP 은닉층 노드수 = 200



transform = transforms.Compose([ #데이터셋 변형 정의. tensor로 변형 후, 평균, 표준편차 0.5로 정규화.
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data_dir = 'celebA/resized_celebA_28/' #데이터셋 경로
dset = dset.ImageFolder(data_dir, transform) #데이터셋 불러오기
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True,drop_last=True) #데이터셋 batch_size만큼 씩 쪼개지게, 섞어서, 마지막 남은샘플 drop.


class Generator(nn.Module): #생성 모델
    def __init__(self): 
        super(Generator,self).__init__() #생성 모델 구조
        self.layer1 = nn.Sequential(OrderedDict([ #첫번째 레이어
                        ('fc1',nn.Linear(z_size,middle_size)), #첫번째 FC 층 z_size에서 middle_size로
                        ('bn1',nn.BatchNorm1d(middle_size)), #batchnorm1d 시행 - 공변량시프트 막기.
                        ('act1',nn.ReLU()), #활성함수 ReLU 통과
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(middle_size,middle_size)), #두번째 FC 층 middle_size에서 middle_size로
                        ('bn1',nn.BatchNorm1d(middle_size)), #batchnorm1d 시행 - 공변량시프트 막기.
                        ('act1',nn.ReLU()), #활성함수 ReLU 통과
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
                        ('fc2', nn.Linear(middle_size,784*3)), #세번째 FC 층 middle_size에서 28*28*3로
                        #('bn2', nn.BatchNorm2d(784)),
                        ('tanh', nn.Tanh()), #활성함수 Tanh통과
        ]))
    def forward(self,z): #전방함수
        out = self.layer1(z) #layer1통과한 out생성
        out = self.layer2(out)#out을 layer2 통과
        out = self.layer3(out)#out을 layer3 통과
        out = out.view(batch_size,3,28,28) #784*3개 출력을 3*28*28로 변형

        return out #out 리턴



class Discriminator(nn.Module): #분별 모델
    def __init__(self):
        super(Discriminator,self).__init__() #분별 모델 구조
        self.layer1 = nn.Sequential(OrderedDict([ #첫번째 레이어
                        ('fc1',nn.Linear(784*3,middle_size)), #첫번째 FC 층 784*3에서 middle_size로
                        #('bn1',nn.BatchNorm1d(middle_size)),
                        ('act1',nn.LeakyReLU()),  #활성함수 LeakyReLU통과
            
        ]))
        self.layer2 = nn.Sequential(OrderedDict([ #두번째 레이어
                        ('fc1',nn.Linear(middle_size,middle_size)), #두번째 FC 층 middle_size에서 middle_size로 
                        #('bn1',nn.BatchNorm1d(middle_size)),
                        ('act1',nn.LeakyReLU()),  #활성함수 LeakyReLU통과
            
        ]))
        self.layer3 = nn.Sequential(OrderedDict([ #세번째 레이어
                        ('fc2', nn.Linear(middle_size,1)), #세번째 FC층 middle_size에서 1개 출력으로
                        #('bn2', nn.BatchNorm2d(1)),
                        ('act2', nn.Sigmoid()), #활성함수 LeakyReLU통과
        ]))
                                    
    def forward(self,x): #전방함수
        out = x.view(batch_size, -1) #batch_size씩 쪼개지는 변형
        #print(out.shape)
        out = self.layer1(out) #out을 layer1 통과
        out = self.layer2(out) #out을 layer2 통과
        out = self.layer3(out) #out을 layer3 통과

        return out #out 리턴


generator = nn.DataParallel(Generator()) #generator 인스턴스 생성 (gpu사용 코드에서 변형)
discriminator = nn.DataParallel(Discriminator()) #discriminator 인스턴스 생성 (gpu사용 코드에서 변형)


loss_func = nn.MSELoss() #Loss 함수 : MSE
#생성 모델 Adam으로 최적화, 학습률은 Learning_rate, 적응모멘텀, 적응학습률 beta=(0.5,0.999)
gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate,betas=(0.5,0.999))
#분별 모델 Adam으로 최적화, 학습률은 Learning_rate, 적응모멘텀, 적응학습률 beta=(0.5,0.999)
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,betas=(0.5,0.999))

ones_label = Variable(torch.ones(batch_size,1)) #batch_size *1크기 1로만 되어있는 label 생성
zeros_label = Variable(torch.zeros(batch_size,1)) #batch_size *1크기 0으로만 되어있는 label 생성

def denorm(x): #정규화된 data 비정규화.
    out = (x + 1) / 2 #(입력+1)/2
    return out.clamp(0, 1) #구간으로 쪼개 출력


fixed_z = Variable(torch.randn(batch_size, z_size)) #비교할 고정된 z값 생성. batch_size, z_size의 랜덤 값 생성
for i in range(epoch): #epoch 만큼 루프
    start_time = time.time() #시작시간 기록
    for j,(image,label) in enumerate(train_loader): #batch_size만큼 받아오면서 루프
         
        image = Variable(image) #dataset에 있는 data를 pytorch variable로 변형
        # discriminator
        
        dis_optim.zero_grad() # 분별 모델 gradient 초기화
        
        z = Variable(init.normal(torch.Tensor(batch_size,z_size),mean=0,std=0.1)) #랜덤 z값 생성.
        gen_fake = generator.forward(z) #z값으로 생성모델 통과해 fake 생성
        dis_fake = discriminator.forward(gen_fake) #생성한 fake를 분별모델 통과해 label 생성
        
        dis_real = discriminator.forward(image) #dataset의 진짜 이미지를 분별모델 통과시켜 label 생성
        #loss값 연산(진짜는 1과, 가짜는 0과 MSE연산 후 합.)
        dis_loss = torch.sum(loss_func(dis_fake,zeros_label)) + torch.sum(loss_func(dis_real,ones_label))
        dis_loss.backward(retain_graph=True)#gradient 연산
        dis_optim.step() #weight 갱신
        
        # generator
        
        gen_optim.zero_grad() #생성모델 gradient 초기화
        
        z = Variable(init.normal(torch.Tensor(batch_size,z_size),mean=0,std=0.1)) #랜덤 z값 생성
        gen_fake = generator.forward(z) #z값으로 생성모델 통과해 fake 생성
        dis_fake = discriminator.forward(gen_fake) #생성한 fake를 분별모델 통과해 lable 생성
        
        gen_loss = torch.sum(loss_func(dis_fake,ones_label)) # fake classified as real
        gen_loss.backward() #gradient 연산
        gen_optim.step() #weight 갱신
    
       
    
        # model save
        if j % 500 == 0: #500번 돌때마다 저장
            print(gen_loss,dis_loss) #loss 출력
            torch.save([generator,discriminator],'./model/vanilla_gan.pkl') #model 저장

            print("{}th iteration gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))
            fake_images=generator.forward(fixed_z) #고정된 z값으로 이미지 생성
            v_utils.save_image(gen_fake.data[0:25],"./result/gen_cur_{}_{}.png".format(i,j), nrow=5)#방금 학습에 사용된 fake 이미지 출력
            v_utils.save_image(denorm(fake_images.data[0:32]), "./result/gen_fix_{}_{}.png".format(i,j), nrow=8) #고정z값으로 연산한 fake 이미지 출력

    print("--- %s seconds ---" %(time.time() - start_time))#epoch 돌때마다 시간 측정.






