import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from scipy.sparse import base
from sklearn.cluster import KMeans
import re
import colorsys
#--------------------------------------처음에는 의류 인식후 의류만 잘라내기를 하려고 했으나 필요없기도 하였고, 불필요하여 주석처리만 함(나중에 공부하기)--------------------


# from tensorflow.keras.models import load_model
# import tensorflow as tf
# import sys

# #https://github.com/anish9/Fashion-AI-segmentation 의류 인식후 잘라내기

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# session = tf.compat.v1.Session(config=config)




# f ="img/up1.jpg" 



# saved = load_model("save_ckp_frozen.h5") #h5라는 파일인데 keras을 통해 CNN등의 딥러닝 모델을 만들어
# #이를 학습시켜 파일로 저장해 놓은것입니다.


# class fashion_tools(object):
#     def __init__(self,imageid,model,version=1.1):
#         self.imageid = imageid
#         self.model   = model
#         self.version = version
        
#     def get_dress(self,stack=False):
        
#         name =  self.imageid
#         file = cv2.imread(name)
#         file = tf.image.resize_with_pad(file,target_height=512,target_width=512)
#         rgb  = file.numpy()
#         file = np.expand_dims(file,axis=0)/ 255.
#         seq = self.model.predict(file)
#         seq = seq[3][0,:,:,0]
#         seq = np.expand_dims(seq,axis=-1)
#         c1x = rgb*seq
#         c2x = rgb*(1-seq)
#         cfx = c1x+c2x
#         dummy = np.ones((rgb.shape[0],rgb.shape[1],1))
#         rgbx = np.concatenate((rgb,dummy*255),axis=-1)
#         rgbs = np.concatenate((cfx,seq*255.),axis=-1)
     
#         if stack:
#             stacked = np.hstack((rgbx,rgbs))
#             return stacked
#         else:
#             return rgbs
                
#     def get_patch(self):
#         return None


# ###running code


# api    = fashion_tools(f,saved)
# image_ = api.get_dress(stack=False)
# cv2.imwrite("out.png",image_)


#-----------------------------------------------------------------------------------------------------------------------------------------




#클러스터의 수를 파악하고 히스토그램을 만듬(각 클러스터에 할당된 픽셀 수를 기반으로)

def centroid_histogram(clt):
  
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # 합이 1이 되도록 히스토그램을 정규화합니다.
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


#-------------------------------------------현재 색상의 빈도에 따라 bar를 만드는 함수------------------------------

def plot_colors(hist, centroids):
    # 각 색상의 상대 빈도를 나타내는 막대 차트 초기화
    
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    maxvalue=0;
    
    # 각 클러스터의 백분율과 색상을 반복합니다.
    #zip은 병렬처리할때 여러 그룹의 데이터를 루프한번만 돌면서 처리 가능하다.(두 그룹의 데이터를 서로 엮어주는 것)
    for (percent, color) in zip(hist, centroids):
        # 각 군집의 상대 백분율을 표시합니다.
        endX = startX + (percent * 300)
        #startX는 bar의 처음부분 endx는 bar의 마지막 좌표부분임 그래서 막대의 크기를 통해 가장 많은색상을 비교해 추출함
        if(maxvalue<(endX-startX)):
            maxvalue=endX
            maxcolor=color
        maxcolor = maxcolor.astype(int)#여기가 가장 많은색상을 추출하는곳
        
        #사각형으로 만듬
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    
    return bar,maxcolor


#-------------------------------------------k값에 따라 bar를 통해 빈도가 높은 색상을 표현------------------------------

def image_color_cluster(image_path,checking, k = 3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    #클러스터 생성 (n_clusters = k)는 정해진 명령어(fit로 클러스터링함)
    clt = KMeans(n_clusters = k)
    clt.fit(image)

    hist = centroid_histogram(clt)
    bar,maxcolor = plot_colors(hist, clt.cluster_centers_)
    if(checking==True):
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
        plt.show()
    
    
    return maxcolor



#-------------------------------------------rgb값을 HSV값으로 바꾸는 함수------------------------------
def convert_rgb_to_hsv(r,g,b):
    #rgb기본 범위: (0-255, 0-255, 0.255)
   

    #get rgb percentage: range (0-1, 0-1, 0-1 )
    red_percentage= r / float(256)
    green_percentage= g/ float(256)
    blue_percentage=b / float(256)

    
    #get hsv percentage: range (0-1, 0-1, 0-1)
    color_hsv_percentage=colorsys.rgb_to_hsv(red_percentage, green_percentage, blue_percentage) 

    
    
    #get normal hsv: range (0-360, 0-255, 0-255)
    color_h=round(360*color_hsv_percentage[0])
    color_s=round(100*color_hsv_percentage[1])
    color_v=round(100*color_hsv_percentage[2])
    color_hsv=(color_h, color_s, color_v)

    return color_hsv

#-----------------------------------체크하는곳------------------------------------------

def check_up_down (up_hsv,down,up_img,down_img):

    
    up_img=cv2.imread(up_img,1)
    down_img=cv2.imread(down_img,1)

    if up_hsv[1]<10 or up_hsv[2]<10:     #무채색은 s값이나 v값이 10보다 크면 색을 얻어서(육안으로) 무채색이 아니게 되므로 10 이하여야한다.
        print("상의가 무채색")
        if down[1]<10 or down[2]<10:  #그냥 상의는무채색인걸 인지하고 무채색의 톤온톤은 무채색이므로 무채색인것만 거르면 톤온톤
            print("무채색 톤온톤 조합 좋음")
            up_img=cv2.resize(up_img,(500,500))
            down_img=cv2.resize(down_img,(500,500))

            add=cv2.vconcat([up_img,down_img])

            cv2.imshow('ton on ton',add)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #무채색의 톤인톤같은 경우는 유채색은 s와 v가 번갈아서 20씩증가하는데 무채색은 v의 값만 13씩 혹은 12씩 증가한다. 그래서 유채색의 톤이 20이하로 떨어지지 않는 점을 고려하여
        #유채색의 s나 v의 값에서 20을 빼고 0.65를 곱하므로써 
        #유채색의 s나 v의 값이 1이 증가함과 무채색의 v값의 증가값을 같게하여
        #그 차이가 3정도의 차이가 날때 톤인톤으로 정한다.
        elif (up_hsv[1]<=52 and abs((down[2]-20)*0.65-up_hsv[2])<5) or up_hsv[1]>52 and abs(((100-down[1])+down[2])*0.65-up_hsv[2])<5 : 
            print("무채색 톤인톤 조합 좋음")
    
            up_img=cv2.resize(up_img,(500,500))
            down_img=cv2.resize(down_img,(500,500))

            add=cv2.vconcat([up_img,down_img])

            cv2.imshow('ton in ton',add)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif down[1]<10: #하의 무채색은 고려해야댐 없애야할지
        print("하의가 무채색")
        if up_hsv[1]<10 or up_hsv[2]<10: #하의가 무채색이면 상의가 무채색이여야 톤온톤이므로 상의 값만 조건으로 둔다
            print("톤온톤 조합 좋음")
            up_img=cv2.resize(up_img,(500,500))
            down_img=cv2.resize(down_img,(500,500))

            add=cv2.vconcat([up_img,down_img])

            cv2.imshow('ton on ton',add)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #하의가 무채색일경우. 상의는 무채색이 무조건 아닌 경우다. 상의가 무채색이면 톤온톤으로 가버리므로 톤인톤이 될 수 없다.
        #하의의 조건의 경우 상의 무채색 하의 유채색의 조건과 반대로 해야한다. 
        # 하의가 무채색인 기준으로 해야한다!
        elif (down[2]<=52 and abs((up_hsv[2]-20)*0.65-down[2])<5) or down[2]>52 and abs(((100-up_hsv[1])+up_hsv[2])*0.65-down[2])<5 :
            print("톤인톤 조합 좋음")
    
            up_img=cv2.resize(up_img,(500,500))
            down_img=cv2.resize(down_img,(500,500))

            add=cv2.vconcat([up_img,down_img])

            cv2.imshow('ton in ton',add)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    #모든 톤온톤은 무채색을 제외하고 0 100 20 <->0 20 100  다음 톤온톤이 29 100 20 <-> 29 20 100 이다.
    #H값은 무조건 같고, s와 v는 20이하로만 가지않으면 톤온톤이다.(20이하는 무채색)
    elif abs(down[0]-up_hsv[0])<30 and (abs(down[1]-up_hsv[1])>=20 or abs(down[2]-up_hsv[2])>=20): 
        print("톤온톤 조합 좋음")  
        up_img=cv2.resize(up_img,(500,500))
        down_img=cv2.resize(down_img,(500,500))

        add=cv2.vconcat([up_img,down_img])

        cv2.imshow('ton on ton',add)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif abs(down[1]-up_hsv[1])<15 and abs(down[2]-up_hsv[2])<15: 
        #톤인톤은 0 100 100 <-> 30 100 100 <->60 100 100 이 톤인톤이 성사된다 
            #그래서 s와 v값은 동일(혹은 15이상[20이상은 다른 톤인톤이 됨])해야하고 h값이 30이면 톤인톤이다.
        print("톤인톤 조합 좋음")   
        up_img=cv2.resize(up_img,(500,500))
        down_img=cv2.resize(down_img,(500,500))

        add=cv2.vconcat([up_img,down_img])

        cv2.imshow('ton in ton',add)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    




#-----------------------------메인부분---체크하는 의류(상의)--------------------------------------
#여기가 내가 선택한 상의를 넣는 곳 상의를 넣게되면 저장되어 있는 하의들과 비교하는 시나리오임
chkimg = "img/up4.png"

image = mpimg.imread(chkimg)
plt.imshow(image)
up_color=image_color_cluster(chkimg,True) #TRUE는 해당 bar를 보여주게끔한다.밑에서는 생략하려고 함
print("체크하는 의류 BRG Format: ",up_color)
colorsB1 =up_color[2]
colorsG1 = up_color[1]
colorsR1 = up_color[0]
up_hsv_color=convert_rgb_to_hsv(colorsR1,colorsG1,colorsB1)
print("저장되는 의류 HSV Format" ,up_hsv_color)



#-----------------------------저장하는 의류(하의)--------------------------------------
img_paths = {
    '검정슬랙스': 'img/down1.jpg',
    '라이트그레이슬랙스': 'img/down2.jpg',
    '카키슬랙스': 'img/down3.jpg',
    '연청바지': 'img/down4.jpg',
    '진청바지': 'img/down5.jpg',
    '생지데님바지': 'img/down6.jpg',
    '데미지청바지': 'img/down7.jpg',
    '회색트레이닝바지':'img/down8.png'
}

 ##npy파일 만들기

descs = {
    '검정슬랙스': None,
    '라이트그레이슬랙스': None,
    '카키슬랙스': None,
    '연청바지': None,
    '진청바지': None,
    '생지데님바지': None,
    '데미지청바지': None,
    '회색트레이닝바지':None
 }

print(type(img_paths))
for name, img_paths in img_paths.items():
    #image = mpimg.imread(img_paths) 만약 저장되는 의류가 보고싶을 떄 삽입코드
    #plt.imshow(image)
    #매개변수 False는 상의의 색상은 사용자가 먼저 볼 수 있게하고, 하의는 너무 많이 있으므로 색상을 보여주는 과정을 생략하기 위함임
    down_color=image_color_cluster(img_paths,False)
   
    #하의의 hsv값을 갖고온다.
    colorsB =down_color[2]
    colorsG = down_color[1]
    colorsR = down_color[0]
    down_hsv_color=convert_rgb_to_hsv(colorsR,colorsG,colorsB)
    #npy파일에 저장하는것을 출력
    print("저장되는 의류 :",name)
    print("저장되는 의류 RGB Format: ", down_color)
    print("저장되는 의류 HSV Format" ,down_hsv_color)
    #상하의의 색상매치하는 함수
    check_up_down (up_hsv_color,down_hsv_color,chkimg,img_paths)

    descs[name] = down_hsv_color
    #각사람의 이름에 맞게 인코딩결과를 저장

np.save('img/descs.npy', descs)
print(descs)


    

