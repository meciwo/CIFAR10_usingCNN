import numpy as np
import pandas as pd

def load_cifar10():

    # 学習データ
    x_train = np.load('/root/userspace/public/lesson2/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson2/data/y_train.npy')

    # テストデータ
    x_test = np.load('/root/userspace/public/lesson2/data/x_test.npy')

    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = np.eye(10)[y_train]

    return (x_train, x_test, y_train)

x_train, x_test, y_train = load_cifar10()

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers

#アンサンブル
pred_y=0

for i in range(5):
    baseMapNum = 32
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(3*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(3*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))


    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    #テスト
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(patience=10, verbose=1)

    #ファイルのロード
    model = load_model('./penpen.h5')

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001), metrics=["accuracy"])
    hist=model.fit(x=x_train, y=y_train, batch_size=32, epochs=20, validation_split=0.1, callbacks=[early_stopping])
    model.save("./penpen.h5")

    pred_y += model.predict(x_test)
ensemble_pred_y = np.argmax(pred_y, 1)
submission = pd.Series(ensemble_pred_y, name='label')
submission.to_csv('/root/userspace/submission.csv', header=True, index_label='id')
