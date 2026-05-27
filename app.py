from flask import *
import os
from werkzeug.utils import secure_filename
import label_image

def load_image(image):
    text = label_image.main(image)
    return text

app = Flask(__name__)

@app.route('/')
@app.route('/first')
def first():
    return render_template('index1.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/model')
def model():
    return render_template('model.html')

 
  
    
@app.route('/login')
def login():
    return redirect(url_for('index'))
@app.route('/chart')
def chart():
    return render_template('chart.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = load_image(file_path)
        result = result.title()
        d = {"Verymild Demented":" → The Very Mild Demented (VMD) stage is an early phase of Alzheimer’s disease and is often considered as part of Mild Cognitive Impairment (MCI). This stage is characterized by subtle symptoms that are usually difficult to detect but can interfere slightly with everyday activities. 1. Memory Problems, 2. Cognitive Challenges, 3.Behavioral Changes, 4.Daily Life Impact",
	'Mild Demented':" → Mild Demented, also referred to as Mild Dementia due to Alzheimer's Disease, is an early stage of dementia where individuals begin to experience noticeable cognitive impairments that affect their daily lives. This stage is crucial as it often marks the progression from normal aging-related memory lapses to more significant challenges. 1. Memory Problems, 2. Cognitive Challenges, 3.Language and Communication Challenges, 4.Mood and Behavior Changes, 5.Loss of Orientation",
        "Moderate Demented":" → Moderate dementia refers to a middle stage of cognitive decline, typically seen in conditions like Alzheimer's disease. In this stage, individuals experience significant memory loss and difficulties in performing everyday tasks, but they may still be able to engage in some level of independent living with assistance. 1. Memory Problems, 2. Cognitive Challenges, 3.Behavioral Changes, 4.Challenges with Daily Activities, 5.Increased Dependence",
        "Non Demented":" → Non-demented refers to individuals who do not exhibit significant cognitive decline or memory impairment severe enough to interfere with daily life. These individuals may still experience normal age-related cognitive changes but remain functionally independent. Understanding this group is crucial in Alzheimer's research and detection, as it helps differentiate between normal aging and early stages of dementia."}
        result = result+d[result]
        #result2 = result+d[result]
        #result = [result]
        #result3 = d[result]        
        print(result)
        #print(result3)
        os.remove(file_path)
        return result
        #return result3
    return None

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
