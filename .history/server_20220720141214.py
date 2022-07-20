from flask import Flask, request,render_template 
import getpass 
import os 
import subprocess
app = Flask(__name__)
app.config.from_object(__name__)

xaxis = [] 
yaxis = [] 
#endpoint 
@app.route('/pre_train', methods=['POST', 'GET'])
def pretraining():
    if(request.method == 'POST'):
        print("Executing Pre-Training scripts")
        import detection
        response = request.get_json()
        #extract filename 
        filename = response.get('value')
        #current os username
        user_name = getpass.getuser() 
        print('accessing the user:' + getpass.getuser())
        image_url = '/Users/' + user_name + '/Desktop/' + filename
        print('the file url is: ' + image_url)
        predictions = detection.detect(image_url)
        
        
        for x in predictions:
             print(x[2]*100)
             xaxis.append(x[2]*100)
        
       
        for y in predictions:
             print(y[1])
             yaxis.append(y[1])
        
        #killing the script to save memory after processing 
        detection.kill()
        #send back the two dimensional array
        return 'done'


@app.route('/re_train', methods=['POST', 'GET'])
def retraining():
        if(request.method == 'POST'):
                print("Executing Re-Training scripts")
                #excute retraining script 
                response = request.get_json()
                filename = response.get('value')
                user_name = getpass.getuser() 
                image_url = '/Users/' + user_name + '/Desktop/' + filename
                python_url = '/Users/' + user_name + '/Desktop/wiseowl/interface/retraining/retrain.py'
                output_pb = '/Users/' + user_name + '/Desktop/wiseowl/output_model/output_graph.pb'
                output_labels ='/Users/' + user_name + '/Desktop/wiseowl/output_model/labels/retrained_label.txt'
                subprocess.Popen(['python3', python_url, '--image' , image_url,  '--graph', output_pb,'--labels' , output_labels ])
                
                return render_template('chart.html', set=zip(xaxis, yaxis))


@app.route('/', methods=['POST', 'GET'])
def front_page():
        if(request.method == 'GET'):
                return render_template('chart.html',set=zip(xaxis, yaxis))

@app.route("/chart")
def chart():
    if(request.method == 'GET'):
            legend = 'pp'
            labels = ["January", "February", "March", "April", "May", "June", "July", "August"]
            values = [10, 9, 8, 7, 6, 4, 7, 8]
            return render_template('chart.html', values=xaxis, labels=yaxis, legend=legend)


if __name__ == '__main__':
    app.run()
