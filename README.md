# clothing-recommendation-system
의류 색상 추출 및 추천 시스템


# 의류 색상 추출 방식

의류 색상 추출은 k-means clustering 을 이용해서 각 픽셀의 가장 빈도수가 많은 값을 추출한다.


![1](https://user-images.githubusercontent.com/90139306/217259625-d2fc218c-203a-4338-960a-9f1856c1e9d3.JPG)


색상 추출은 RGB를 추출하고 RGB값을 HSV값으로 변환시킨다.


![2](https://user-images.githubusercontent.com/90139306/217259630-ac3ead99-1e16-477f-9675-2f6538dd8e13.JPG)


HSV값이 값을 숫자로 표현했을때의 규칙이 톤온톤 톤인톤에 적합하여 선택했다.


![3](https://user-images.githubusercontent.com/90139306/217259633-65b6f16f-32b5-4c05-b12a-8c8d58c5f763.JPG)



# 추천 방식

Ton On Ton 과 Ton In Ton을 기준으로 추천한다.


![4](https://user-images.githubusercontent.com/90139306/217259637-9fc6b881-2ee2-4c42-ad29-6a8a60905ad0.JPG)


# 추천 시스템 예시


![5](https://user-images.githubusercontent.com/90139306/217259640-b855ab1f-7510-48fb-9ef7-9472cae10f48.JPG)


# 실행 과정


![6](https://user-images.githubusercontent.com/90139306/217259641-ce68af7e-2dba-4f16-93c8-f666eab262e5.JPG)


![11](https://user-images.githubusercontent.com/90139306/217259642-15779cf8-dd22-428c-81e8-4f8adf52ecc4.JPG)
