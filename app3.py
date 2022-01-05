import streamlit as st
import tensorflow as tf
import plotly.express as px
import sys
from io import StringIO
from PIL import Image
from plotly.subplots import make_subplots

class Page:
    def __init__(self):
        st.set_page_config(page_title = 'Deep Learning com Tensorflow & Keras')

    def page(self):
        st.title('Predições com redes neurais pré-treinadas')
        st.caption(
            """Caso você acesse alguma dessas páginas em um smartphone, 
            use no modo paisagem (horizontal) para que as imagens fiquem em tamanho correto."""
        )
        with st.expander('1. Rede neural pré-treinada VGG16', expanded = True):
            st.header('Rede neural pré-treinada VGG16')
            st.write(
                """**VGG16 é um modelo de rede neural convolucional proposta por K. Simonyan e A. Zisserman,
                ambos da Universidade de Oxford, no artigo "Very Deep Convolutional Networks for Large-Scale
                Image Recognition". O modelo obteve 92.7% de acurácia de teste no conjunto de imagens ImageNet,
                entre os cinco modelos com maior acurácia para tal conjunto.**""")
            st.write("""**ImageNet é um conjunto de dados com mais de 14 milhões de imagens pertencentes a 1000 classes.**""")
            st.caption('Fonte: https://neurohive.io/en/popular-networks/vgg16/')
            self.vgg16_img1 = tf.keras.utils.get_file('vgg16-1-e1542731207177.png',
                                                    'https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png',
                                                    extract = True)
            st.image(self.vgg16_img1, caption = 'Estrutura da rede neural VGG16')
            self.vgg16_img2 = tf.keras.utils.get_file('vgg16.png',
                                                        'https://neurohive.io/wp-content/uploads/2018/11/vgg16.png',
                                                        extract = True)
            st.image(self.vgg16_img2, caption = 'Estrutura da rede neural VGG16')

        with st.expander('2. Estrutura da rede neural', expanded = True):
            st.header('Estrutura da rede neural')
            st.subheader('Resumo da rede neural')
            self.model = tf.keras.applications.vgg16.VGG16(weights = 'imagenet', include_top = True)
            #####Escrevendo o output do resumo #####
            old_stdout = sys.stdout = mystdout = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            st.text(mystdout.getvalue())
            ########################################

        with st.expander('3. Fazendo predições', expanded = True):
            st.header('Fazendo predições')
            st.caption('Envie uma imagem para o modelo de rede neural classificar.')
            st.caption('Você pode arrastar uma imagem diretamente do Google Imagens para esta caixa de upload.')
            self.file = st.file_uploader('Envie uma imagem', 
                                        type = ['png', 'jpg'],
                                        accept_multiple_files = False)
            if self.file is not None:
                self.file_img = Image.open(self.file)
                self.tensor_img = tf.convert_to_tensor(self.file_img)
                self.fig_file1 = px.imshow(self.tensor_img)
                self.tensor_reshaped = tf.image.resize(self.tensor_img, (224,224))
                self.fig_file2 = px.imshow(self.tensor_reshaped)
                self.subplots_files = make_subplots(rows = 1, cols = 2)
                self.subplots_files.add_trace(self.fig_file1.data[0], row = 1, col = 1)
                self.subplots_files.add_trace(self.fig_file2.data[0], row = 1, col = 2)
                self.subplots_files.update_xaxes(showticklabels = False, 
                                            title_text = 'Tamanho original (shape): {}'.format(self.tensor_img.shape),
                                            row = 1,
                                            col = 1)
                self.subplots_files.update_yaxes(showticklabels = False,
                                            row = 1,
                                            col = 1)
                self.subplots_files.update_xaxes(showticklabels = False, 
                                            title_text = 'Tamanho redimensionado (shape): {}'.format(self.tensor_reshaped.shape),
                                            row = 1,
                                            col = 2)
                self.subplots_files.update_yaxes(showticklabels = False,
                                            row = 1,
                                            col = 2)
                st.plotly_chart(self.subplots_files)
                st.caption('A imagem é redimensionada para fluir pela rede neural e ser classificada.')
                st.subheader('Compilando o modelo')
                self.optim = st.selectbox('Escolha um otimizador para a rede neural',
                                            ['SGD', 'Adam', 'RMSprop'])
                self.compile = st.button('Compilar modelo')
                if self.compile:
                    self.model.compile(optimizer = self.optim,
                                        loss = 'categorical_crossentropy')
                    st.write('---')
                    self.tensor_pred = tf.expand_dims(self.tensor_reshaped, axis = 0)
                    self.tensor_pred = tf.keras.applications.imagenet_utils.preprocess_input(self.tensor_pred)
                    self.probabilities = self.model.predict(self.tensor_pred)
                    self.P = tf.keras.applications.imagenet_utils.decode_predictions(self.probabilities)
                    st.subheader('Predição')
                    st.caption('Este modelo costuma identificar o objeto presente na imagem.')
                    st.write('Classe predita: {}'.format(self.P[0][0][1]))
                    st.subheader('Probabilidades')
                    self.col40, self.col41, self.col42, self.col43, self.col44 = st.columns(5)
                    with self.col40:
                        st.metric(self.P[0][0][1], '{:.2f}%'.format(self.P[0][0][2] * 100))
                    with self.col41:
                        st.metric(self.P[0][1][1], '{:.2f}%'.format(self.P[0][1][2] * 100))
                    with self.col42:
                        st.metric(self.P[0][2][1], '{:.2f}%'.format(self.P[0][2][2] * 100))
                    with self.col43:
                        st.metric(self.P[0][3][1], '{:.2f}%'.format(self.P[0][3][2] * 100))
                    with self.col44:
                        st.metric(self.P[0][4][1], '{:.2f}%'.format(self.P[0][4][2] * 100))

        with st.expander('Código Fonte'):
            st.header('Código Fonte')
            self.python_file = open('app3.py', 'r').read()
            st.code(self.python_file, language = 'python')


page = Page().page()