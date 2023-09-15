import streamlit as st
import requests
import json
import io


def post_run(image_bytes):
    url = "http://backend_fastapi:5000/predict"
    #url = "http://localhost:5000/predict"
    payload = {}
    files=[
    ('file',('image.JPG',image_bytes,'application/octet-stream'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    return (response.text)

def run():
    st.write("# CHEST X RAY CLASSIFIER :doctor:")

    st.write("### this web app can classify 15 commonly happening disease in lungs and heart using uploaded image of the chest x ray :doctor:")

    st.write('upload the file to analyze the health of your patient:point_down:')

   


    # create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # check if an image file was uploaded
    if uploaded_file is not None:
        # display the uploaded image
        st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
        #uploaded_file = UploadedFile(uploaded_file.name, uploaded_file.type, uploaded_file.size, uploaded_file.data)
    
    if st.button('Predict health'):
        image_bytes = uploaded_file.read()
        prediction_response = post_run(image_bytes)
        
        # Parse the JSON string to a dictionary
        try:
            prediction = json.loads(prediction_response)
        except json.JSONDecodeError:
            st.write("Error parsing the response.")
            return
        
        # Debug: Print the whole response to ensure it's what you expect
        #st.write("Complete Response:", prediction)

        predicted_diseases = prediction.get('predictions_with_prob', {})

        # Display the results
        st.write("Predicted Class:", prediction.get('class', 'N/A'))
        st.write("threshold set :0.28")
        st.write("Predicted probabilities for each disease:")
        for disease, probability in predicted_diseases.items():
            st.write(f"{disease}: {probability * 100:.2f}%")

if __name__ == '__main__':
    run()