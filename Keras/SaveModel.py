from keras.models import Sequential
import os 

model = Sequential()

target_dir = './models/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)


model.save('./models/model.h5')
model.save_weights('./models/weights.h5')