import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import plotly.express as px
from PIL import Image
from plotly.subplots import make_subplots

class Page:
    def __init__(self):
        st.set_page_config(page_title = 'Deep Learning com Tensorflow & Keras')

    def page(self):
        st.title('Aplicação do Tensorflow Hub')
        st.write('_Autor: Rafael Silva Coelho_')
        st.caption(
            """Caso você acesse alguma dessas páginas em um smartphone, 
            use no modo paisagem (horizontal) para que as imagens fiquem em tamanho correto."""
        )
        st.write(
            """Muitas vezes, criar e treinar uma rede neural do zero é um processo demorado, que pode levar
            horas ou até mesmo dias, devido a limitação de hardware da máquina que processa a rede neural,
            o tamanho da rede, o tamanho do conjunto de dados entre outros fatores.  
            Nesse sentido, é mais viável usar uma rede já treinada para fazer novas aplicações. É o que
            faremos nessa seção."""
        )

        with st.expander('1. Transferência de estilo', expanded = True):
            st.header('Transferência de estilo')
            st.write(
                """**O trabalho original de transferência artística de estilo com redes neurais propôs 
                um algoritmo de otimização lento que funciona com qualquer pintura arbitrária. Um trabalho 
                subsequente desenvolveu um método para transferência de estilo rápida que pode funcionar em 
                tempo real, mas é limitado para um ou para um conjunto limitado de estilos**"""
            )
            st.caption('Fonte: https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            st.write('---')
            st.subheader('Exemplo de transferência de estilo')
            self.subplots_transfer = make_subplots(rows = 1, cols = 2)
            self.image1 = tf.keras.utils.load_img('image1.jpg')
            self.fig_im1 = px.imshow(self.image1)
            self.image2 = tf.keras.utils.load_img('image2.jpg')
            self.fig_im2 = px.imshow(self.image2)
            self.subplots_transfer.add_trace(self.fig_im1.data[0], row = 1, col = 1)
            self.subplots_transfer.add_trace(self.fig_im2.data[0], row = 1, col = 2)
            self.subplots_transfer.update_xaxes(showticklabels = False,
                                                title_text = 'Foto de Rafael',
                                                row = 1,
                                                col = 1)
            self.subplots_transfer.update_xaxes(showticklabels = False,
                                                title_text = 'Estilo a ser aplicado',
                                                row = 1,
                                                col = 2)
            self.subplots_transfer.update_yaxes(showticklabels = False)
            st.plotly_chart(self.subplots_transfer)
            ##### Montando a transferência de estilo
            self.content_image1 = tf.cast(self.image1, tf.float32) / 255
            self.content_image1 = tf.expand_dims(self.content_image1, axis = 0)
            self.style_image1 = tf.cast(self.image2, tf.float32)
            self.style_image1 = tf.image.resize(self.style_image1, (256,256))
            self.style_image1 = tf.expand_dims(self.style_image1, axis = 0) / 255
            #A imagem de conteúdo pode ter qualquer tamanho
            #A imagem de estilo deve ter tamanho 256x256
            self.hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            self.stylized_image = tf.keras.utils.load_img('image3.jpeg')
            self.stylized_fig = px.imshow(self.stylized_image)
            self.stylized_fig.update_xaxes(showticklabels = False, title_text = 'Resultado final da transferência de estilo')
            self.stylized_fig.update_yaxes(showticklabels = False)
            st.plotly_chart(self.stylized_fig)
            st.write('---')
            st.subheader('Faça você mesmo')
            self.col50, self.col51 = st.columns(2)
            with self.col50:
                self.content_image2 = st.file_uploader('Envie uma foto', 
                                                        type = ['jpg', 'png'], 
                                                        accept_multiple_files = False)
                if self.content_image2 is not None:
                    self.content_image_plot = Image.open(self.content_image2)
                    self.content_image2 = tf.cast(self.content_image_plot, tf.float32) / 255
                    self.content_image2 = tf.expand_dims(self.content_image2, axis = 0)
            with self.col51:
                self.style_image2 = st.file_uploader('Envie um desenho ou alguma arte', 
                                                        type = ['jpg', 'png'], 
                                                        accept_multiple_files = False)
                if self.style_image2 is not None:
                    self.style_image_plot = Image.open(self.style_image2)
                    self.style_image2 = tf.image.resize(self.style_image_plot, (256, 256))
                    self.style_image2 = tf.cast(self.style_image2, tf.float32) / 255
                    self.style_image2 = tf.expand_dims(self.style_image2, axis = 0)
            st.caption('O resultado final será processado assim que você enviar as duas imagens.')
            st.caption('Você pode arrastar imagens diretamente do Google Imagens para estas duas caixas de upload.')
            if (self.content_image2 is not None) and (self.style_image2 is not None):
                with st.spinner('Carregando o resultado. Aguarde.'):
                    self.outputs2 = self.hub_module(tf.constant(self.content_image2), tf.constant(self.style_image2))
                self.stylized_image2 = self.outputs2[0][0]
                self.download_image = tf.cast(self.stylized_image2 * 255, tf.uint8)
                #self.download_image = Image.fromarray(self.download_image.numpy()).save('stylized_image.png')
                self.final_subplots = make_subplots(rows = 1, cols = 3)
                self.content_fig = px.imshow(self.content_image_plot)
                self.style_fig = px.imshow(self.style_image_plot)
                self.final_fig = px.imshow(self.stylized_image2)
                self.final_subplots.add_trace(self.content_fig.data[0], row = 1, col = 1)
                self.final_subplots.add_trace(self.style_fig.data[0], row = 1, col = 2)
                self.final_subplots.add_trace(self.final_fig.data[0], row = 1, col = 3)
                self.final_subplots.update_xaxes(showticklabels = False)
                self.final_subplots.update_xaxes(title_text = 'Conteúdo', row = 1, col = 1)
                self.final_subplots.update_xaxes(title_text = 'Estilo', row = 1, col = 2)
                self.final_subplots.update_xaxes(title_text = 'Conteúdo estilizado', row = 1, col = 3)
                self.final_subplots.update_yaxes(showticklabels = False)
                st.plotly_chart(self.final_subplots)
                st.image(self.download_image.numpy(), caption = 'Resultado Final')

        with st.expander('Código Fonte'):
            st.header('Código Fonte')
            self.python_file = open('app4.py', 'r').read()
            st.code(self.python_file, language = 'python')

page = Page().page()
