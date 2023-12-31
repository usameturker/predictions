import streamlit as st
import pandas as pd
import pickle


def page1():
    st.header("Sales Forecast based on Page Engagement")

    st.image('https://www.revechat.com/wp-content/uploads/2016/09/Website-Engagement.jpg', use_column_width=True)

    st.title("Please select the page engagement values below:")

    usp = st.number_input("What is the unit session percentage?", 0.00, 1.00, step=0.01)
    pw = st.number_input("What is the total page views?", 0, 10000, step=10)
    ppw = st.number_input("What is the total percentage of page views?", 0.00, 1.00, step=0.01)


    my_dict = {
        "unit_session_percentage": usp,
        "page_views_total": pw,
        "page_views_percentage_total": ppw
    }

    df = pd.DataFrame.from_dict([my_dict])

    st.subheader("Your Product's Page Engagement")
    st.dataframe(df)

    model = pickle.load(open("rf_model_final", "rb"))

    prediction = model.predict(df)

    if st.button('PREDICT'):

        if prediction>0:

            st.success("Sales expected", icon="✅")
        else:

            st.error('Sales NOT expected', icon="❌")

def page2():
    
    import tempfile
    import os
    from urllib.request import urlopen

    st.title("Category Prediction of Images")
    
    st.image('https://static.thestudentroom.co.uk/cms/sites/default/files/2023-04/prediction%20article.png',use_column_width=True)

    uploaded_file = st.file_uploader("Upload an image from your PC", type=["jpg", "png", "jpeg"])
    image_url = st.text_input("Or paste the URL of an image")

    from roboflow import Roboflow

    rf = Roboflow(api_key="A0HwSSkoqIrnz9hxlrBs")
    project = rf.workspace().project("category")
    model = project.version(1).model

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())

        st.image(temp_file.name, caption="Uploaded Image", width=300)

        jsonfile = model.predict(temp_file.name).json()
        try:
            predicted_classes = jsonfile["predictions"][0]["predicted_classes"]
            
            st.subheader("Predicted Category:")
            
            if len(predicted_classes) == 0:
                st.error('Could not Predicted')
            elif len(predicted_classes) == 1:
                st.success(predicted_classes[0], icon="✅")
            elif len(predicted_classes) == 2:
                if predicted_classes[0]>predicted_classes[1]:
                    st.success(predicted_classes[0], icon="✅")
                else:
                    st.success(predicted_classes[1], icon="✅")
            
            
            
        except IndexError as e:
            st.error('This image could not be predicted.')

        os.unlink(temp_file.name) 

    elif image_url:
        st.image(image_url, caption="Image from URL", width=300)

        with urlopen(image_url) as response:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(response.read())

        jsonfile = model.predict(temp_file.name).json()
        
        try:
            predicted_classes = jsonfile["predictions"][0]["predicted_classes"]
            
            st.subheader("Predicted Category:")
            
            if len(predicted_classes) == 0:
                st.error('Could not Predicted')
            if len(predicted_classes) == 1:
                st.success(predicted_classes[0], icon="✅")
            elif len(predicted_classes) == 2:
                if predicted_classes[0]>predicted_classes[1]:
                    st.success(predicted_classes[0], icon="✅")
                else:
                    st.success(predicted_classes[1], icon="✅")
            
            
            
        except IndexError as e:
            st.error('This image could not be predicted.')
        

        os.unlink(temp_file.name) 


st.sidebar.image('https://oneamz.com/wp-content/uploads/2021/10/logo-black.8bf86065-e1633441524735.png')        
selected_page = st.sidebar.radio("Select the Prediction Model", ("Sales Prediction", "Category Prediction"))


if selected_page == "Sales Prediction":
    page1()
elif selected_page == "Category Prediction":
    page2()
