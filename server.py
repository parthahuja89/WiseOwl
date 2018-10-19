from flask import Flask, request 

app = Flask(__name__)
app.config.from_object(__name__)

#endpoint 
@app.route('/pre_train', methods=['POST', 'GET'])
def pretraining():
    if(request.method == 'POST'):
        import detection
        reponse = request.get_json()
        image_url = 'strawberry.jpg'
        predictions = detection.detect(image_url)
        xaxis = []
        for x in predictions:
             print(x)
             xaxis.append(x[2]*100)

        yaxis = []
        for y in predictions:
             print(y)
             yaxis.append(y[1])

        #send back the two dimensional array 
if __name__ == '__main__':
    app.run()
