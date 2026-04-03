from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import os
import gdown
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "biscuit_model_v2.tflite"
MODEL_URL = "https://drive.google.com/uc?id=1AzGm3DhynGS0IXp1ZzsCOsPNsiMiWsTd"

if not os.path.exists(MODEL_PATH):
    print("Downloading TFLite model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [
    "Americana Coconut Cookies", "Amul Chocolate Cookies", "Amul Elaichi Rusk", "Bhagwati Choco Vanilla Puff Biscuits", "Bhagwati Lemony Puff Biscuits",
    "Bisk Farm Sugar Free Biscuits", "Bonn Jeera Bite Biscuits", "Britannia 50-50 Maska Chaska", "Britannia 50-50 Maska Chaska Salted Biscuits",
    "Britannia 50-50 Potazos - Masti Masala", "Britannia 50-50 Sweet and Salty Biscuits", "Britannia 50-50 Timepass Classic Salted Biscuit",
    "Britannia Biscafe Coffee Cracker", "Britannia Bourbon", "Britannia Bourbon The Original Cream Biscuits", "Britannia Chocolush - Pure Magic",
    "Britannia Good Day - Chocochip Cookies", "Britannia Good Day Cashew Almond Cookies", "Britannia Good Day Harmony Biscuit",
    "Britannia Good Day Pista Badam Cookies", "Britannia Little Hearts", "Britannia Marie Gold Biscuit", "Britannia Milk Bikis Milk Biscuits",
    "Britannia Nice Time - Coconut Biscuits", "Britannia Nutri Choice Oats Cookies - Chocolate and Almonds",
    "Britannia Nutri Choice Oats Cookies - Orange With Almonds", "Britannia Nutri Choice Seed Biscuits", "Britannia Nutri Choice Sugar Free Cream Cracker Biscuits",
    "Britannia Nutrichoice Herbs Biscuits", "Britannia Tiger Glucose Biscuit", "Britannia Tiger Kreemz - Chocolate Cream Biscuits",
    "Britannia Tiger Kreemz - Elaichi Cream Biscuits", "Britannia Tiger Kreemz - Orange Cream Biscuits", "Britannia Tiger Krunch Chocochips Biscuit",
    "Britannia Treat Chocolate Cream Biscuits", "Britannia Treat Crazy Pineapple Cream Biscuit", "Britannia Treat Jim Jam Cream Biscuit",
    "Britannia Treat Osom Orange Cream Biscuit", "Britannia Vita Marie Gold Biscuits", "Cadbury Bournvita Biscuits", "Cadbury Chocobakes Choc Filled Cookies",
    "Cadbury Oreo Chocolate Flavour Biscuit Cream Sandwich", "Cadbury Oreo Strawberry Flavour Creme Sandwich Biscuit",
    "Canberra Big Orange Cream Biscuits", "CookieMan Hand Pound Chocolate Cookies", "Cremica Coconut Cookies", "Cremica Elaichi Sandwich Biscuits", "Cremica Jeera Lite",
    "Cremica Non-Stop Thin Potato Crackers - Baked, Crunchy Masala", "Cremica Orange Sandwich Biscuits", "Krown Black Magic Cream Biscuits", "MARIO Coconut Crunchy Biscuits",
    "McVities Bourbon Cream Biscuits", "McVities Dark Cookie Cream", "McVities Marie Biscuit", "Parle 20-20 Cashew Cookies",
    "Parle 20-20 Nice Biscuits", "Parle Happy Happy Choco-Chip Cookies", "Parle Hide and Seek", "Parle Hide and Seek - Black Bourbon Choco",
    "Parle Hide and Seek - Milano Choco Chip Cookies", "Parle Hide and Seek Caffe Mocha Cookies", "Parle Hide and Seek Chocolate and Almonds",
    "Parle Krack Jack Original Sweet and Salty Cracker Biscuit", "Parle Krackjack Biscuits", "Parle Magix Sandwich Biscuits - Chocolate",
    "Parle Milk Shakti Biscuits", "Parle Monaco Biscuit - Classic Regular", "Parle Monaco Piri Piri", "Parle Platina Hide and Seek Creme Sandwich - Vanilla",
    "Parle-G Gold Gluco Biscuits", "Parle-G Original Gluco Biscuits", "Patanjali Doodh Biscuit", "Priyagold Butter Delite Biscuits",
    "Priyagold CNC Biscuits", "Priyagold Cheese Chacker Biscuits", "Priyagold Snacks Zig Zag Biscuits", "Richlite Rich Butter Cookies",
    "RiteBite Max Protein 7 Grain Breakfast Cookies - Cashew Delite", "Sagar Coconut Munch Biscuits", "Sri Sri Tattva Cashew Nut Cookies",
    "Sri Sri Tattva Choco Hazelnut Cookies", "Sri Sri Tattva Coconut Cookies", "Sri Sri Tattva Digestive Cookies",
    "Sunfeast All Rounder - Cream and Herb", "Sunfeast All Rounder - Thin, Light and Crunchy Potato Biscuit With Chatpata Masala Flavour",
    "Sunfeast Bounce Creme Biscuits", "Sunfeast Bounce Creme Biscuits - Elaichi", "Sunfeast Bounce Creme Biscuits - Pineapple Zing",
    "Sunfeast Dark Fantasy - Choco Creme", "Sunfeast Dark Fantasy Bourbon Biscuits", "Sunfeast Dark Fantasy Choco Fills",
    "Sunfeast Glucose Biscuits", "Sunfeast Moms Magic - Fruit and Milk Cookies", "Sunfeast Moms Magic - Rich Butter Cookies",
    "Sunfeast Moms Magic - Rich Cashew and Almond Cookies", "Tasties Chocochip Cookies", "Tasties Coconut Cookies",
    "UNIBIC Choco Chip Cookies", "UNIBIC Pista Badam Cookies", "UNIBIC Snappers Potato Crackers"
]

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        input_tensor = preprocess(image)

        # Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        prediction_index = np.argmax(output, axis=1)[0]
        predicted_class = labels[prediction_index] if prediction_index < len(labels) else "Unknown"
        confidence = np.max(output) * 100
        result = f"{predicted_class} ({confidence:.2f}% confidence)"

        return render_template("index.html", prediction=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
