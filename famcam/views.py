from django.shortcuts import render, redirect, reverse
from common.FaceCapture import CapFace

# Create your views here.
def famcam(request):
    error = None
    if request.method == 'POST':
        photo_path = request.POST.get('fpath')
        famYN = request.POST.get('famYn')
        selSRC = request.POST.get('selSRC')
        print(photo_path, famYN, selSRC)
        if famYN == 'F':
            if selSRC == 'C':
                cap_face_from_cam(request)
            else:
                if photo_path is None or photo_path=='':
                    error = '얼굴을 추출할 가족 이미지 폴더를 입력하세요'
                else:
                    cap_face_from_img(request, photo_path, famYN)
        else:
            if photo_path is None or photo_path=='':
                # display an error message
                error = '얼굴을 추출할 이미지 폴더를 입력하세요'
            else:
                cap_face_from_img(request, photo_path, famYN)

    return render(request, 'famcam/fammain.html')

def cap_face_from_cam(request):
    cf = CapFace('caffe')
    cf.cap_face_from_cam()

    return render(request, 'famcam/fammain.html')

def cap_face_from_img(request, path, famYN):
    cf = CapFace('caffe')
    cf.captureFromImg(path, famYN)

    return render(request, 'famcam/fammain.html')

def learnphoto(request):
    pass

def startcam(request):
    pass