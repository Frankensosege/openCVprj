from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16

def TrainModel():
    train_data_gen = ImageDataGenerator(rescale=1./255,            # 정규화
                                       horizontal_flip=True,       # 수평 뒤집기
                                       fill_mode='nearest')
    train_generator = train_data_gen.flow_from_directory('./data/train', target_size=(150, 150), batch_size=5, class_mode='binary')

    test_data_gen = ImageDataGenerator(rescale=1./255)  # 테스트 데이터는 정규화만
    test_generator = test_data_gen.flow_from_directory('./data/test', target_size=(150, 150), batch_size=5, class_mode='binary')

    ## transfer learning
    transfer_vgg16 = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3))
    transfer_vgg16.trainable = False

    finetune_vgg16 = Sequential()
    finetune_vgg16.add(transfer_vgg16)
    finetune_vgg16.add(Flatten())
    finetune_vgg16.add(Dense(64, activation='relu'))
    finetune_vgg16.add(Dropout(0.3))
    finetune_vgg16.add(Dense(1, activation='sigmoid'))

    finetune_vgg16.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    checkpoint = ModelCheckpoint(filepath = './data/model/reco_family.h5', save_best_only=True)
    erarly_stop = EarlyStopping(patience=10)

    history = finetune_vgg16.fit(train_generator, epochs=100, batch_size = 20, validation_data=test_generator, callbacks=[checkpoint, erarly_stop])
