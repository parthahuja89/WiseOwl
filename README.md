# WiseOwl 
A image recogntion app that uses pre-trained(all images) and re-trained (on eagles and owls) machine learning models. 
# Installation 
1) Clone rep on desktop (only works on desktop dir) 
2) Download retrained model on eagles and owls: output_graph.pb: https://drive.google.com/file/d/1_rrL3KjcbcHLGKfjMjxxrvYz9yHzQJtk/view?usp=sharing
3) Paste the model in /output_model/
4) Run the python server in the /wiseowl-master/ ---  python server.py
5) Run Electron app in /Inteface/ --</br>
                                   |-- npm install </br>
                                   |-- npm run start</br>
6) Paste target image in Desktop, and select image through UI.

# HomeScreen 
![alt text](https://github.com/parthahuja89/WiseOwl/blob/master/test_images/homescreen.png)

# Testing Image 
![alt text](https://github.com/parthahuja89/WiseOwl/blob/master/test_images/owl.jpeg)

# Pre-Trained Output
#### 90% confidence. 
![alt text](https://github.com/parthahuja89/WiseOwl/blob/master/test_images/Pre-Trained.png)

# Re-Trained Output
#### 85% confidence. 
![alt text](https://github.com/parthahuja89/WiseOwl/blob/master/test_images/Retrained.png)
