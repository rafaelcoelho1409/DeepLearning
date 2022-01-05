import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.express as px
import sys
from io import StringIO

class Page:
    def __init__(self):
        st.set_page_config(page_title = 'Deep Learning com Tensorflow & Keras')

    def page(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train_reshaped = self.x_train.reshape(60000, 28 * 28).astype('float32') / 255
        self.x_test_reshaped = self.x_test.reshape(10000, 28 * 28).astype('float32') / 255
        self.y_train_reshaped = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test_reshaped = tf.keras.utils.to_categorical(self.y_test, 10)
        
        st.title('Redes Neurais Simples - MNIST')
        st.caption(
            """Caso você acesse alguma dessas páginas em um smartphone, 
            use no modo paisagem (horizontal) para que as imagens fiquem em tamanho correto."""
        )
        with st.expander('1. Informações sobre o conjunto de dados MNIST', expanded = True):
            st.header('Informações sobre o conjunto de dados MNIST')
            st.write(
                """**O conjunto de dados MNIST é composto por várias imagens de dígitos de 0 a 9
                escritos a mão.**""")
            self.col00, self.col01 = st.columns(2)
            with self.col00:
                st.write('Tamanho do tensor de dados de imagens de treino:')
                st.write(self.x_train_reshaped.shape)
                st.caption('60000 imagens de tamanho 28x28 achatadas em vetores de tamanho 784 = 28*28')
                st.write('Tamanho do tensor de rótulos de imagens de treino:')
                st.write(self.y_train_reshaped.shape)
                st.caption('60000 rótulos convertidos em 0 e 1')
            with self.col01:
                st.write('Tamanho do tensor de dados de imagens de teste:')
                st.write(self.x_test_reshaped.shape)
                st.caption('10000 imagens de tamanho 28x28 achatadas em vetores de tamanho 784 = 28*28')
                st.write('Tamanho do tensor de rótulos de imagens de teste:')
                st.write(self.y_test_reshaped.shape)
                st.caption('10000 rótulos convertidos em 0 e 1')
            st.write('---')
            ##### Número de amostras #####
            self.frame_train = pd.DataFrame(self.y_train).rename(columns = {0: 'rótulo'})
            self.frame_train['contagem'] = 1
            self.frame_train = self.frame_train.groupby('rótulo').sum().reset_index()
            self.frame_train['tipo'] = 'treino'
            self.frame_test = pd.DataFrame(self.y_test).rename(columns = {0: 'rótulo'})
            self.frame_test['contagem'] = 1
            self.frame_test = self.frame_test.groupby('rótulo').sum().reset_index()
            self.frame_test['tipo'] = 'teste'
            self.frame = pd.concat([self.frame_train, self.frame_test])
            ##############################
            self.fig = px.bar(self.frame, 
                        x = 'tipo', 
                        y = 'contagem', 
                        color = 'rótulo', 
                        title = 'Número de amostras por rótulo e tipo de conjunto (treino ou teste)',
                        barmode = 'group',
                        text = 'rótulo')
            st.plotly_chart(self.fig)
            st.caption('O conjunto é levemente desbalanceado em relação a cada classe.')

        with st.expander('2. Visualização das imagens do conjunto de dados', expanded = True):
            st.header('Visualização das imagens do conjunto de dados')
            self.container = st.container()
            self.index = st.number_input('Índice da imagem; Anterior (-) | Próxima (+)', 
                                        value = 0, 
                                        min_value = 0, 
                                        max_value = self.x_train.shape[0] - 1,
                                        step = 1)
            self.image = px.imshow(self.x_train[int(self.index)])
            self.image.update_coloraxes(showscale=False)
            self.image.update_xaxes(showticklabels = False, 
                                    title_text = 'Rótulo: {}'.format(self.y_train[int(self.index)]))
            self.image.update_yaxes(showticklabels = False)
            self.container.plotly_chart(self.image)

        with st.expander('3. Estruturando a rede neural', expanded = True):
            st.header('Estruturando a rede neural')
            ##### Montando as camadas da rede neural #####
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Dense(64, 
                                                input_shape = (self.x_train_reshaped.shape[1],), 
                                                activation = 'relu'))
            self.model.add(tf.keras.layers.Dense(32,
                                        activation = 'relu'))
            self.model.add(tf.keras.layers.Dense(10, 
                                        activation = 'softmax'))
            st.subheader('Resumo da rede neural')
            #############################################
            #####Escrevendo o output do resumo #####
            old_stdout = sys.stdout = mystdout = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            st.text(mystdout.getvalue())
            ########################################
            st.write('---')
            ##### Otimizadores e hiperparâmetros #####
            st.subheader('Treinando a rede neural')
            self.optimizer = st.selectbox('Escolha o otimizador da rede neural',
                                    ['Stochastic Gradient Descent (SGD)', 'RMSprop', 'Adam'])
            if self.optimizer == 'Stochastic Gradient Descent (SGD)':
                self.optim = tf.keras.optimizers.SGD()
            elif self.optimizer == 'RMSprop':
                self.optim = tf.keras.optimizers.RMSprop()
            elif self.optimizer == 'Adam':
                self.optim = tf.keras.optimizers.Adam()
            self.loss = tf.keras.losses.CategoricalCrossentropy()
            self.model.compile(optimizer = self.optim, 
                                loss = self.loss, metrics = 'accuracy')
            self.train_nn = st.button('Treinar rede neural')
            if self.train_nn:
                with st.spinner('Treinando rede neural. Aguarde.'):
                    self.history = self.model.fit(self.x_train_reshaped, 
                                                    self.y_train_reshaped,
                                                    epochs = 20,
                                                    batch_size = 500,
                                                    validation_data = (self.x_test_reshaped, self.y_test_reshaped))
                    #acurácia por época (treino e teste)
                    self.train_accuracy = pd.DataFrame(self.history.history['accuracy']).reset_index().rename(columns = {'index': 'épocas', 0: 'acurácia'})
                    self.train_accuracy['tipo'] = 'treino'
                    self.test_accuracy = pd.DataFrame(self.history.history['val_accuracy']).reset_index().rename(columns = {'index': 'épocas', 0: 'acurácia'})
                    self.test_accuracy['tipo'] = 'teste'
                    self.accuracy = pd.concat([self.train_accuracy, self.test_accuracy])
                    #perda / taxa de erro por época (treino e teste)
                    self.train_loss = pd.DataFrame(self.history.history['loss']).reset_index().rename(columns = {'index': 'épocas', 0: 'perda'})
                    self.train_loss['tipo'] = 'treino'
                    self.test_loss = pd.DataFrame(self.history.history['val_loss']).reset_index().rename(columns = {'index': 'épocas', 0: 'perda'})
                    self.test_loss['tipo'] = 'teste'
                    self.loss = pd.concat([self.train_loss, self.test_loss])
                    #acurácia de teste
                    self.final_test_loss, self.final_test_acc = self.model.evaluate(self.x_test_reshaped, self.y_test_reshaped)
                    self.col10, self.col11 = st.columns(2)
                    with self.col10:
                        st.metric('Acurácia de teste', '{:.2f}%'.format(self.final_test_acc * 100))
                    with self.col11:
                        st.metric('Erro de teste', '{:.2f}'.format(self.final_test_loss))
                    #plotando acurácia e erro
                    self.fig1 = px.line(self.accuracy,
                                    x = 'épocas', 
                                    y = 'acurácia',
                                    color = 'tipo',
                                    title = 'Acurácia por época')
                    self.fig1.layout.yaxis.tickformat = ',.0%'
                    self.fig2 = px.line(self.loss,
                                    x = 'épocas',
                                    y = 'perda',
                                    color = 'tipo',
                                    title = 'Perda / taxa de erro por época')
                    st.plotly_chart(self.fig1)
                    st.plotly_chart(self.fig2)

        with st.expander('Código Fonte'):
            st.header('Código Fonte')
            self.python_file = open('app1.py', 'r').read()
            st.code(self.python_file, language = 'python')


page = Page().page()
