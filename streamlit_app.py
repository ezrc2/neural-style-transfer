import streamlit as st
import nst
from PIL import Image

st.set_page_config(page_title='Neural Style Transfer')
st.title('Neural Style Transfer')

st.write(f'Device: {nst.get_device()}')

content_image_file = st.file_uploader(label='Content Image:', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
style_image_file = st.file_uploader(label='Style Image:', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
st.caption('Default content image is Mona Lisa, default style image Starry Night.')

if content_image_file is None:
    content_image_file = 'images/mona_lisa.jpeg'
if style_image_file is None:
    style_image_file = 'images/starry_night.jpeg'

content_image = nst.loader(Image.open(content_image_file))
style_image = nst.loader(Image.open(style_image_file))

epochs = st.slider(label='Number of epochs:', min_value=500, max_value=1500, step=100)
st.caption('Start with 600 to 800 epochs. If the result image is unclear, increase the number of epochs.')

b = st.button(label='Run')
if b:
    result_images = nst.fit(content_image=content_image, style_image=style_image, epochs=epochs)
    result = result_images[-1]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write('Content Image')
        st.image(nst.to_output(content_image))
    with c2:
        st.write('Style Image')
        st.image(nst.to_output(style_image))
    with c3:
        st.write('Result Image')
        st.image(result)