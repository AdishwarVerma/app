import streamlit as st
from PIL import Image
import pandas as pd

from io import BytesIO,StringIO
import streamlit.components.v1 as stc
import numpy as np
import json
import urllib
import matplotlib.pyplot as plt
import os



from yaya import mortality
import shap
from shap import Explanation
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Mortality rate of an ICU",page_icon=":tada:",layout="wide")

title_alignment="""
<style>
#the-title {
  text-align: center
  max-width : 50%
  left-padding : 30%
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)

st.title("                                   Patient Mortality Analyser")




def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)
lt=load_lottiefile("110260-online-doctor.json")
st_lottie(lt,loop=True
          ,height=200)

# doc=st.text_input("Enter Your Name Respected Doctor")
# result=st.button("Next")
# if result:
#     st.write("Welcome Dr "+ doc+ "Hope you are having an amazing day")

STYLE = """
<style>
img {
     max-width :100%;
}
</style>
"""
st.subheader("Please Insert The Patient's Data: ")

# st.info(__doc__)
st.markdown(STYLE, unsafe_allow_html=True)
show_fie = st.empty()


show_fie.info("Please upload a  file in the following format: {}".format(' '.join(["csv" ",","txt"])))
#file=st.file_uploader("Upload file" ,type=["csv","txt"])

file = st.file_uploader("Upload file", type=["csv", "txt"])



if file is not None and file.size!=0:



















        zz=mortality(file)
        #ss=str(zz[0])

        st.write("The probility of survival is = ", zz[0]
                 )



        st.set_option('deprecation.showPyplotGlobalUse', False)
        # kk=new_func(file)

        with st.container():
            st.write("---")
            left_column ,right_column = st.columns(2)
            with left_column:

                st.subheader("Features Impact On Patient")
                kkk = zz[2]
                ddd = shap.plots.waterfall(kkk)
                plt.xlabel("Percentage")
                plt.ylabel("Features")

                plt.legend()
                st.pyplot(ddd)
            with right_column:
                kk=zz[1]
                lst = kk.values.tolist()[0]
                new_inds1=round(max(lst),2)
                new_inds2=round(min(lst),2)
                new_ind1 = lst.index(max(lst))
                new_ind2=lst.index(min(lst))


                index_of = kk.feature_names[new_ind1]
                index_ofs=kk.feature_names[new_ind2]
                #st.write("The main variable affecting the probability is ", index_of)
                st.write("The most impactful feature for this patient that is working against its survival is ",index_of ,", with an increase of=",str(new_inds1))
                st.write("The most impactful feature for this patient that is working infavour its survival is ", index_ofs,", with an increase of=", str(new_inds2))

        # with st.container():
        #     st.write("---")
        #     left_column, right_column = st.columns(2)
        #     with left_column:
        #
        #
        #
        #         st.subheader("cmndasbjchsj,hfgKSDVJ,Hasdgf,")
        #         kk=zz[1]
        #
        #         dd=shap.plots.beeswarm(kk)
        #         st.pyplot(dd)
        #     with right_column:
        #         kk=zz[1]
        #         lst=kk.values.tolist()[0]
        #         new_ind=lst.index(max(lst))
        #
        #         index_of=kk.feature_names[new_ind]
        #         st.write("The main variable affecting the probability is ",index_of)
        #st.write(kk.feature_names[vals])


        with st.container():
            st.write("----")

            left_column, right_column = st.columns(2)
            with left_column:
                st.subheader("Feature Impact On Model")

                bars=shap.summary_plot(kk, plot_type="bar")
                plt.xlabel("Percentage")
                plt.ylabel("Features")


                st.pyplot(bars)





            with right_column:
                st.write("The bar plot given on the left shows the feature importance of the shap model.")




else:
    st.write("Error Please enter a valid file ")