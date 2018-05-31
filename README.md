# ASRUSI
Attention-based Sentence Reconstruction that Utilizes the Spacing Information (for Korean)

## Requirements
KoNLPy, pyfasttext, Keras (TensorFlow), Numpy, NLTK

## Word Vector 
https://drive.google.com/file/d/1PE7RWIjLyBaBdrLg_ybUpnoJAvibkrXY/view?usp=sharing
* Download this and unzip THE FOLDER in the same folder with 'fxcute.py' 
* Loading the model will be processed by load_model('vector/model')

## System Description
* Easy start: Python3 execute file
<pre><code> python3 fxcute.py </code></pre>
* This system reconstructs a full sentence from a sequence of morphemes
- ex) 잘 했 어 넌 못 참 았 을 거야 >> "잘 했어 넌 못 참았을거야"
