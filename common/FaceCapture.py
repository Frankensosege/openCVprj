from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import decode_predictions
import sys
import cv2
from  common.kakaotalk import KatalkApi
import glob
import datetime

class CapFace:
    def __init__(self, model):
        if model == 'caffe':
            ## caffe
            model = './ssd_faceDetector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
            config = './ssd_faceDetector/deploy.prototxt.txt'
        else:
            ## tensorflow
            model = './ssd_faceDetector/opencv_face_detector_uint8.pb'
            config = './ssd_faceDetector/opencv_face_detector.pbtxt.txt'

        self.fampath = './data/train/family/'
        self.othpath = './data/train/not/'

        self.face_net = cv2.dnn.readNet(model, config)
        if self.face_net.empty():
            print('Model read failed!!!!!!!!')
            sys.exit()

    def cap_face_from_cam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print('Camera open failed')
            sys.exit()

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print('frame with failed')
                break

            blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 117, 123), swapRB=False)
            self.face_net.setInput(blob)
            outs = self.face_net.forward()

            detect = outs[0, 0, :, :]
            h, w = frame.shape[:2]

            for i in range(detect.shape[0]):
                confidence = detect[i, 2]
                if confidence > 0.5:
                    x1 = int(detect[i, 3] * w)
                    y1 = int(detect[i, 4] * h)
                    x2 = int(detect[i, 5] * w)
                    y2 = int(detect[i, 6] * h)
                    text = f'{confidence * 100:.2f}%'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                    file_name = 'face_{0:04d}.jpg'.format(idx)
                    idx += 1
                    save_img = frame[y1:y2, x1:x2, :]
                    cv2.imwrite(self.fampath + file_name, save_img)

            cv2.imshow('image', frame)
            key = cv2.waitKey(20)
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()

    def captureFromImg(self, path, photo_type):
        img_paths = glob.glob(path + '/*.*')

        if photo_type=='F':
            save_path = self.fampath
        else:
            save_path = self.othpath

        idx = 0
        for img_path in img_paths:
            img = cv2.imread(img_path)
            print(img_path)
            #     print(img.shape)
            #     break
            blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 117, 123), swapRB=False)
            self.face_net.setInput(blob)
            outs = self.face_net.forward()

            detect = outs[0, 0, :, :]
            h, w = img.shape[:2]

            save_img = []
            for i in range(detect.shape[0]):
                confidence = detect[i, 2]
                if confidence > 0.5:
                    x1 = int(detect[i, 3] * w)
                    y1 = int(detect[i, 4] * h)
                    x2 = int(detect[i, 5] * w)
                    y2 = int(detect[i, 6] * h)

                    file_name = 'face_{0:04d}.jpg'.format(idx)
                    idx += 1
                    save_img = img[y1:y2, x1:x2, :]
                    cv2.imwrite(save_path + file_name, save_img)

    def play_cam(self):
        kakao = KatalkApi()
        kakao.getJson()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print('Camera open failed')
            sys.exit()
        model = load_model('./data/model/reco_family.hdf5')
        next_time = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print('frame with failed')
                break

            ## 입력 이미지 정제
            blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 117, 123), swapRB=False)
            self.face_net.setInput(blob)
            ## 결과
            outs = self.face_net.forward()

            detect = outs[0, 0, :, :]
            h, w = frame.shape[:2]

            for i in range(detect.shape[0]):
                confidence = detect[i, 2]
                if confidence > 0.5:
                    x1 = int(detect[i, 3] * w)
                    y1 = int(detect[i, 4] * h)
                    x2 = int(detect[i, 5] * w)
                    y2 = int(detect[i, 6] * h)

                    prd_img = frame[y1:y2, x1:x2, :].copy()
                    # X_test = np.expand_dims(prd_img[0], axis=0)
                    pred = model.predict(prd_img)
                    classnames = decode_predictions(pred)
                    print(classnames)
                    if pred <= 0.5:
                        text = f'{confidence * 100:.2f}%'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                        if next_time <= datetime.datetime.now():
                            kakao('침입자 발생')
                            next_time = datetime.datetime.now() + datetime.timedelta(hours=1)

            cv2.imshow('image', frame)
            key = cv2.waitKey(20)
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()