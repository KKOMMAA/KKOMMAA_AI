from flask import Flask
from flask import request 
import tensorflow as tf
import os
import transformers
from tensorflow_addons.optimizers import RectifiedAdam


app = Flask(__name__)


def secure_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"
    return "".join(safe_char(c) for c in s).rstrip("_")



# stt 텍스트를 받아 감정값 긍정 or 부정을 판별하는 api <진>

#지금은 이미지 입력해 감정값 나오는 ai 모델
@app.route("/model", methods=['POST']) 
def model():
    if request.method == 'POST':
        print(request.json)
        print(request.json['content'])
        input = request.json['content']

        # file = request.files['image'] #파일 받기
        # file_path = 'dataset/'+secure_filename(file.filename)
        # print(file_path)
        # file.save(file_path) #로컬에 저장
        print(os.getcwd())
        #model = tf.keras.models.load_model(os.getcwd()+'/sentiment_model1.h5')

        tf.keras.optimizers.RectifiedAdam = RectifiedAdam
        model = tf.keras.models.load_model(os.getcwd()+'/sentiment_model1.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

        model.summary()
        prediction = model.predict((input,'0','0')) #input을 넣기
        
        print("예상값! : "+ prediction)

        
        result = prediction

    return jsonify({'emotion': prediction}), 200



if __name__ == '__main__':
    app.run()