import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.express as px
import sys
import time
from io import StringIO
from plotly.subplots import make_subplots

class Page:
    def __init__(self):
        st.set_page_config(page_title = 'Deep Learning com Tensorflow & Keras')

    def page(self):
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
                - Barcdeeplearning/dl_streamlit.pyo  
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
            self.datagen_plot = tf.keras.preprocessing.image.ImageDataGenerator(
                                                                        rotation_range = 30,
                                                                        width_shift_range = 0.2,
                                                                        height_shift_range = 0.2,
                                                                        horizontal_flip = True)
            self.datagen_plot.fit(self.x_train)
            #OBS: foram usados dois geradores de imagens. O datagen será usado para os dados normalizados,
            #que serão usados na rede neural. O datagen_plot é apenas para mostrar as transformações nas imagens,
            #ou seja, meramente ilustrativo.
            ##1000 batches (lotes) de 50 amostras
            self.batches = self.datagen.flow(self.x_train_norm, self.y_train_reshaped, batch_size = 100)
            self.batches_plot = self.datagen_plot.flow(self.x_train, self.y_train_reshaped, batch_size = 200)
            #st.write(self.batches[0][0][0].shape)
            self.batches_subplots = make_subplots(rows = 3, cols = 3)
            for i in range(9):
                globals()['batch_image_{}'.format(i)] = self.batches_plot[0][0][i]
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
            self.example_image = self.batches_plot[0][0][0]
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
                globals()['imshow1_{}'.format(i)] = px.imshow(im[0,:,:,:3],)
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
                ou de um cachorro, ou de um gato etc."""
            )

        with st.expander('6. Treinando a rede neural e verificando acurácia', expanded = True):
            st.header('Treinando a rede neural e verificando acurácia')
            self.optim = st.selectbox('Escolha o otimizador da rede neural', ['SGD', 'Adam', 'RMSprop'])
            st.caption('Número de épocas: 2')
            st.caption('Número de images por lote: 100')
            st.caption('A rede neural leva aproximadamente 5 minutos para ser treinada.')
            self.model.compile(loss = 'categorical_crossentropy', 
                                optimizer = self.optim, 
                                metrics = ['accuracy'])
            self.train_nn = st.button('Treinar rede neural')
            if self.train_nn:
                with st.spinner('Treinando rede neural. Aguarde.'):
                    self.start = time.time()
                    self.history = self.model.fit(self.batches, 
                                            epochs = 2, 
                                            validation_data = (self.x_test_norm, self.y_test_reshaped))
                    self.end = time.time()
                    st.write('O tempo de treino da rede neural foi de {:.2f} segundos'.format(self.end - self.start))
                    self.col30, self.col31, self.col32, self.col33 = st.columns(4)
                    with self.col30:
                        st.metric('Acurácia de treino', '{:.2f}%'.format(self.history.history['accuracy'][0] * 100))
                    with self.col31:
                        st.metric('Erro de treino', '{:.2f}'.format(self.history.history['loss'][0]))
                    with self.col32:
                        st.metric('Acurácia de teste', '{:.2f}%'.format(self.history.history['val_accuracy'][0] * 100))
                    with self.col33:
                        st.metric('Erro de teste', '{:.2f}'.format(self.history.history['val_loss'][0]))
                    st.write('Quanto maior for a rede, quanto mais épocas e quanto mais lotes tiver seu conjunto de dados, '
                            'mais demorado será o treino, porém muito mais acertos sua rede terá. Neste caso, a rede neural construída '
                            'teve baixa acurácia por ser curta e ter apenas uma época, para não demorar muito. Esta construção é '
                            'apenas para fins ilustrativos e didáticos.')
                    st.write('Em geral, usa-se redes neurais já treinadas e salvas para se fazer predições, quando a aplicação '
                            'não exige constante atualização.')

        with st.expander('Código Fonte'):
            st.header('Código Fonte')
            self.python_file = open('app2.py', 'r').read()
            st.code(self.python_file, language = 'python')


page = Page().page()