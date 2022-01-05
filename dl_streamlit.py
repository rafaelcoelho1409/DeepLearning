import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import plotly.express as px
import sys
import time
from io import StringIO
from PIL import Image
from plotly.subplots import make_subplots


class MultiPage:
    def __init__(self):
        st.set_page_config(page_title = 'Deep Learning com Tensorflow & Keras')
        self.sidebar_title = st.sidebar.title('Deep Learning com Tensorflow & Keras')
        self.sidebar_pages = st.sidebar.radio('', 
                                                ['Redes Neurais Simples - MNIST',
                                                 'Redes Neurais Convolucionais - CNNs',
                                                 'Predições com redes neurais pré-treinadas',
                                                 'Aplicação do Tensorflow Hub',
                                                 'Código Python deste projeto'])
        if self.sidebar_pages == 'Redes Neurais Simples - MNIST':
            self.page_1()
        elif self.sidebar_pages == 'Redes Neurais Convolucionais - CNNs':
            self.page_2()
        elif self.sidebar_pages == 'Predições com redes neurais pré-treinadas':
            self.page_3()
        elif self.sidebar_pages == 'Aplicação do Tensorflow Hub':
            self.page_4()
        elif self.sidebar_pages == 'Código Python deste projeto':
            self.page_5()
    
    def page_1(self):
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

    def page_2(self):
        st.title('Redes Neurais Convolucionais - CNNs')
        st.caption(
            """Caso você acesse alguma dessas páginas em um smartphone, 
            use no modo paisagem (horizontal) para que as imagens fiquem em tamanho correto."""
        )
        with st.expander('1. Informações sobre o conjunto de dados CIFAR-10', expanded = True):
            st.header('Informações sobre o conjunto de dados CIFAR-10')
            st.write(
                """**O conjunto de imagens CIFAR-10 é constituído de várias imagens dos seguintes objetos:**  
                - Avião  
                - Automóvel  
                - Pássaro  
                - Gato  
                - Cervo  
                - Cachorro  
                - Sapo  
                - Cavalo  
                - Barco  
                - Caminhão""")
            ##### Importando o conjunto de dados
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_test.astype('float32')
            # Normalizando os dados
            self.mean = tf.reduce_mean(self.x_train)
            self.std = tf.math.reduce_std(self.x_train)
            self.x_train_norm = (self.x_train - self.mean)/(self.std)
            self.x_test_norm = (self.x_test - self.mean)/(self.std)
            self.y_train_reshaped = tf.keras.utils.to_categorical(self.y_train, 10) #10 classes
            self.y_test_reshaped = tf.keras.utils.to_categorical(self.y_test, 10)
            st.write('---')
            self.col20, self.col21 = st.columns(2)
            with self.col20:
                st.write('Tamanho do tensor de dados de imagens de treino')
                st.write(self.x_train_norm.shape)
                st.caption('50000 imagens de tamanho 32x32 em 3 canais de cores (RGB)')
                st.write('Tamanho do tensor de rótulos de imagens de treino')
                st.write(self.y_train_reshaped.shape)
            with self.col21:
                st.write('Tamanho do tensor de dados de imagens de teste')
                st.write(self.x_test_norm.shape)
                st.caption('10000 imagens de tamanho 32x32 em 3 canais de cores (RGB)')
                st.write('Tamanho do tensor de rótulos de imagens de teste')
                st.write(self.y_test_reshaped.shape)
            ##### Número de amostras
            self.frame_train = pd.DataFrame(self.y_train).rename(columns = {0: 'rótulo'})
            self.frame_train['contagem'] = 1
            self.frame_train = self.frame_train.groupby('rótulo').sum().reset_index()
            self.frame_train['tipo'] = 'treino'
            self.frame_test = pd.DataFrame(self.y_test).rename(columns = {0: 'rótulo'})
            self.frame_test['contagem'] = 1
            self.frame_test = self.frame_test.groupby('rótulo').sum().reset_index()
            self.frame_test['tipo'] = 'teste'
            self.frame = pd.concat([self.frame_train, self.frame_test])
            #########################
            self.fig = px.bar(self.frame, 
                        x = 'tipo', 
                        y = 'contagem',
                        color = 'rótulo',
                        title = 'Número de amostras por rótulo e tipo de conjunto (treino ou teste)',
                        barmode = 'group',
                        text = 'rótulo')
            st.plotly_chart(self.fig)
            st.caption('O conjunto é balanceado em relação a todas as suas classes')

        with st.expander('2. Visualização das imagens do conjunto de dados', expanded = True):
            st.header('Visualização das imagens do conjunto de dados')
            self.subplots1 = make_subplots(rows = 2, cols = 5)
            self.labels = {0: 'Avião',
                            1: 'Carro',
                            2: 'Pássaro',
                            3: 'Gato',
                            4: 'Cervo',
                            5: 'Cachorro',
                            6: 'Sapo',
                            7: 'Cavalo',
                            8: 'Barco',
                            9: 'Caminhão'}
            self.y_labels = pd.DataFrame(self.y_train).rename(columns = {0: 'rótulo'})
            self.container2 = st.container()
            self.index2 = st.number_input('Índice da imagem; Anterior (-) | Próxima (+)', 
                                        value = 0, 
                                        min_value = 0, 
                                        max_value = self.x_train.shape[0] - 1,
                                        step = 1)
            self.image2 = px.imshow(self.x_train[int(self.index2)])
            self.image2.update_coloraxes(showscale=False)
            self.image2.update_xaxes(showticklabels = False, 
                                    title_text = 'Rótulo: {}'.format(self.labels[self.y_train[int(self.index2)][0]]))
            self.image2.update_yaxes(showticklabels = False)
            self.container2.plotly_chart(self.image2)

        with st.expander('3. Aumento de imagens do conjunto', expanded = True):
            st.header('Aumento de imagens do conjunto')
            st.write(
                """**Usando a aplicação ImageDataGenerator do Keras, é possível criar novas imagens
                a partir das imagens já existentes, fazendo rotações, espelhamentos,
                ampliação ou redução da imagem etc.**""")
            st.write("""**O intuito desta aplicação é melhorar
                a acurácia do modelo (quanto mais amostras, mais o modelo aprende, 
                logo existirão mais acertos).**""")
            st.write(
                """As imagens abaixo foram produzidas a partir do ImageDataGenerator."""
            )
            self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                                        rotation_range = 30,
                                                                        width_shift_range = 0.2,
                                                                        height_shift_range = 0.2,
                                                                        horizontal_flip = True)
            self.datagen.fit(self.x_train_norm)
            self.batches = self.datagen.flow(self.x_train_norm, self.y_train_reshaped, batch_size = 100)
            self.batches_subplots = make_subplots(rows = 3, cols = 3)
            for i in range(9):
                globals()['batch_image_{}'.format(i)] = (self.batches[0][0][i] * self.std) + self.mean
                globals()['batch_fig_{}'.format(i)] = px.imshow(globals()['batch_image_{}'.format(i)])
                self.batches_subplots.add_trace(globals()['batch_fig_{}'.format(i)].data[0],
                                                row = int(i/3)+1,
                                                col = i%3+1)
                self.batches_subplots.update_xaxes(showticklabels = False)
                self.batches_subplots.update_yaxes(showticklabels = False)
            st.plotly_chart(self.batches_subplots)

        with st.expander('4. Estruturando a rede neural', expanded = True):
            st.header('Estruturando a rede neural')
            st.write('**Abaixo, você visualiza o que acontece com a imagem em cada camada da rede neural.**')
            self.example_image = self.batches[0][0][0]
            #1st block
            self.input = tf.keras.layers.Input(shape = self.x_train_norm.shape[1:])
            self.conv1_0 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu')
            self.x = self.conv1_0(self.input)
            self.bn1_0 = tf.keras.layers.BatchNormalization()
            self.x = self.bn1_0(self.x)
            self.conv1_1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu')
            self.x = self.conv1_1(self.x)
            self.bn1_1 = tf.keras.layers.BatchNormalization()
            self.x = self.bn1_1(self.x)
            self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size = (2,2))
            self.x = self.maxpool1(self.x)
            self.dropout1 = tf.keras.layers.Dropout(0.2)
            self.x = self.dropout1(self.x)
            #dense block
            self.flatten = tf.keras.layers.Flatten()
            self.x = self.flatten(self.x)
            self.dense = tf.keras.layers.Dense(10, activation = 'softmax')
            self.output = self.dense(self.x)
            #model
            self.model = tf.keras.Model(inputs = self.input, outputs = self.output)
            #####Escrevendo o output do resumo #####
            st.subheader('Resumo da rede neural')
            old_stdout = sys.stdout = mystdout = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            st.text(mystdout.getvalue())
            ########################################

        with st.expander('5. O que acontece por dentro da rede neural?', expanded = True):
            st.header('O que acontece por dentro da rede neural?')
            st.write(
                """OBS: para cada camada, foram usadas apenas 3 canais de cores, pois para cada camada de convolução,
                o número de camadas de cores salta (3 > 32 > 64 > 128). Por exemplo, o tensor da imagem tem tamanho (32,32,3),
                e ao passar pela primeira camada de convolução, passa a ter tamanho (32,32,32)."""
            )
            self.example_image = (self.batches[0][0][0] * self.std) + self.mean
            #o tensor precisa ter 4 dimensões para fluir pela rede neural
            self.example_image = tf.expand_dims(self.example_image, axis = 0)
            #1st block
            self.im1_0 = self.conv1_0(self.example_image)
            self.im1_1 = self.bn1_0(self.im1_0)
            self.im1_2 = self.conv1_1(self.im1_1)
            self.im1_3 = self.bn1_1(self.im1_2)
            self.im1_4 = self.maxpool1(self.im1_3)
            self.im1_5 = self.dropout1(self.im1_4)
            #plotando
            self.subplots = make_subplots(rows = 3, cols = 6)
            self.im1 = [self.im1_0, self.im1_1, self.im1_2, self.im1_3, self.im1_4, self.im1_5]
            for i, im in enumerate(self.im1):
                globals()['imshow1_{}'.format(i)] = px.imshow(im[0,:,:,:3])
            for i in range(6):
                self.subplots.add_trace(globals()['imshow1_{}'.format(i)].data[0], row = 1, col = i + 1)
            self.example_plot = px.imshow(self.example_image[0])
            self.example_plot.update_xaxes(showticklabels = False, title_text = 'Imagem original')
            self.example_plot.update_yaxes(showticklabels = False)
            self.subplots.update_xaxes(showticklabels = False)
            self.subplots.update_yaxes(showticklabels = False)
            self.subplots.update_xaxes(title_text = 'Conv2D', row = 1, col = 1)
            self.subplots.update_xaxes(title_text = 'BatchNorm', row = 1, col = 2)
            self.subplots.update_xaxes(title_text = 'Conv2D', row = 1, col = 3)
            self.subplots.update_xaxes(title_text = 'BatchNorm', row = 1, col = 4)
            self.subplots.update_xaxes(title_text = 'MaxPooling2D', row = 1, col = 5)
            self.subplots.update_xaxes(title_text = 'Dropout', row = 1, col = 6)
            self.subplots.update_layout(title = 'Camadas da rede neural')
            st.plotly_chart(self.example_plot)
            st.plotly_chart(self.subplots)
            st.write(
                """Após passar por todas estas camadas em sequência, o tensor é transformado em um vetor unidimensional
                (camada Flatten), e a partir disso, são camadas de perceptrons (Multilayer Perceptrons) para se obter a
                predição de qual classe aquela imagem pertence (camada Dense), ou seja, se aquela imagem é de um avião,
                ou de um cachorro, ou de um gato etc.""")
            
    def page_3(self):
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
                        
    def page_4(self):
        st.title('Aplicação do Tensorflow Hub')
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
            with st.spinner('Carregando o resultado'):
                self.outputs = self.hub_module(tf.constant(self.content_image1), tf.constant(self.style_image1))
            self.stylized_image = self.outputs[0][0]
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

    def page_5(self):
        st.title('Código Python da página')
        self.python_file = open('dl_streamlit.py', 'r').read()
        st.code(self.python_file, language = 'python')

multipage = MultiPage()
