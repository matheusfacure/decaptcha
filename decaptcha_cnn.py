# -*- coding: utf-8 -*- 
from glob import glob # para achar os arquivos
from PIL import Image # para manipular as imagens
import numpy as np # para trabalhar com dados de maneira eficiente
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split # para validação cruzada
import os # para salvar o progresso
import tensorflow as tf # para redes neurais
from time import time # para cronometrar
from scipy import ndimage # para manipular imagens
from sys import argv, exit # para execução na linha de comando


#####################################
####### Hiper-Parâmetros ############
#####################################

# Hiper-parametros de treinamento
learning_rate = 1e-3 # taxa de aprendizado
keep_prob_train = 0.45 # probabilidade de manter os dados no dropout
gamma = 0.0005 # fator de regularização dos parâmetros
batch_size = 32 # tamanho do mini-lote
max_delta = 0.5 # quantidade de luminosidade máxima para ajustar
rotate = True # se rotaciona os mini-lotes de treino (aumenta bastante o tempo de treino)
rot_degree = 20 # quantidade máxima de graus para rotacionar a imagem
restore = True # para resumir o treinamento de um checkpoint ou comecar do zero

# Quantidade de treinamento
training_iters = 30000 # iterações de treinamento
display_step = 3000 # regularidade para mostrar resultados
min_val_acc = 72 # acurácia mínima no set de validação para salvar
show_learning = True # plota o custo a cada iteração

# Hiper-parametros da Rede Neural (se manipulados, terá que treinar do zero)
img_h = 36 # altura da imagem depois de cortar (deve ser multiplo de 4)
img_w = 120 # largura da imagem depois de cortar (deve ser múltiplo de 4)
CL1_depth = 16 # profundidade do kernel da primeira camada convolucioal
CL2_depth = 32 # profundidade do kernel da segunda camada convolucional
CL3_depth = 64 # profundidade do kernel da terceira camada convolucional
DL1_size = 128 # tamanho da primeira camada densa
DL2_size = 128 # tamanho da segunda camada densa
n_classes = 11  # quantidade de saida (10 dígitos mais nulo)
seq_len = 6 # tamanho da máxima sequência de dígitos



#####################################
########## Sub-rotinas ##############
#####################################

# cria diretório de checkpoints da rede neural
save_dir = 'DECAPCHAcheckpoints/'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

# função para carregar os dados
def get_data(n_data):
	img_dir = 'train_imgs/' # diretório ccom as imagens
	img_files = glob(img_dir + '*.png')
	img_files = img_files[:min(len(img_files), n_data)] # lista com todas as imagens
	
	image_data = []
	image_label = []
	for img_file in img_files:

		filename = img_file[len(img_dir):] # acha o nome da imagem
		
		# cria os dados (X)
		im = Image.open(img_file).convert('L') # carrega a imagem em preto e branco
		imgx = np.array(im.getdata()) # convere imagem em array Numpy
		imgx = (imgx - imgx.mean()) / imgx.std() # normaliza imagem
		image_data.append(imgx.reshape(40, 140)) # formata a imagem e junta aos dados
		
		# cria as etiquetas (y)
		label_str = filename[:filename.find('.')]
		
		# acha o nome numérico da imagem
		if label_str.find(' ') != -1: # se for a segunda imagem com o mesmo digito, haverá ' '.
			label_str = label_str[:label_str.find(' ')]
		if label_str.find('(') != -1: # se for a segunda imagem com o mesmo digito, pode haver '('.
			label_str = label_str[:label_str.find('(')]
		
		img_oh_label = np.zeros((6,11)) # cria array vazio para preencher como vetores one-hot
		
		for digit in range(6): # tamanho máximo da sequencia é de 6 digitos
			if digit < len(label_str):
				img_oh_label[digit-len(label_str), int(label_str[digit])] = 1.0 # coloca um no espaço do dígito
			else:
				img_oh_label[digit-len(label_str), 10] = 1.0 # preenche slotes vazios
		
		image_label.append(img_oh_label)
		
	return np.array(image_data), np.array(image_label)


def get_test_data(img_files):

	image_data = []
	for img_file in img_files:

		# cria os dados (X)
		im = Image.open(img_file).convert('L') # carrega a imagem em preto e branco
		imgx = np.array(im.getdata()) # convere imagem em array Numpy 
		imgx = (imgx - imgx.mean()) / imgx.std() # normaliza imagem
		image_data.append(imgx.reshape(40, 140))

	return np.array(image_data)


# camada convolucional
def conv2d(img, w, b, name):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w,
						strides=[1, 1, 1, 1], padding='SAME'), b), name=name)

# camada de max_pool com down-sample de fator 2
def max_pool(img, name, kk=2, ks=2):
	return tf.nn.max_pool(img, ksize=[1, kk, kk, 1],
						strides=[1, ks, ks, 1], padding='SAME', name=name)

# calcula a acurácia
def accuracy(pred_y, true_y):
	pred_labels = np.argmax(pred_y, 2)
	true_labels = np.argmax(true_y, 2)
	return  np.all(pred_labels == true_labels, 1).mean() * 100

# softmax çoputacionalmente estável (para n ter problema de overflow em exp())
def stable_softmax(x):
	if len(x.shape) > 1:
		tmp = np.max(x, axis = 1)
		x -= tmp.reshape((x.shape[0], 1))
		x = np.exp(x)
		tmp = np.sum(x, axis = 1)
		x /= tmp.reshape((x.shape[0], 1))
	
	else:
		tmp = np.max(x)
		x -= tmp
		x = np.exp(x)
		tmp = np.sum(x)
		x /= tmp
	
	return x

# converte a previsão do modelo em digitos do captcha
def sample_digit(y_hat):
	captcha_solved = [] 
	for digit in y_hat:

		# converte os scores em probabilidades válidas
		softmax = stable_softmax(digit) - 0.0005
		
		# sorteia o dígito segundo a distribição condicional aprendida
		sampled = np.random.multinomial(1, softmax)

		number = str(sampled.argmax())
		if number != '10': # ignora nulos
			captcha_solved.append(number)

	return ''.join(captcha_solved)



# funções para aumento artificial de dados
# rotaciona a imagem em um angulo aleatório entre -6 e 6
def rotate_batch(imgs):
	graus = np.random.randint(-rot_degree,rot_degree)
	return ndimage.rotate(imgs, graus, mode='nearest', axes=(1,2), reshape=False)   

def augment(image):
	image = tf.random_crop(image, size=[img_h, img_w, 1]) # ajusta a luminosidade aleatoriamente
	image = tf.image.random_brightness(image, max_delta, seed=None) # corta a img aleatoriamente
	return image

def center_crop(image):
	image = tf.image.resize_image_with_crop_or_pad(image, img_h, img_w) # corta a imagem no centro
	return image




run_flags = argv[1:] # argumentos da linha de comando



###################################################
####### Construção do grafo TensorFlow ############
###################################################

################ inputs do grafo ##################
tf_x_input = tf.placeholder(tf.float32, [None, 40, 140], name='X_input')
tf_X = tf.reshape(tf_x_input, shape=[-1, 40, 140, 1]) # reformata input para forma de imagem requerida pelo tf (h, w, d)

tf_y_input = tf.placeholder(tf.int64, [None, seq_len, n_classes], name='y_input')
tf_keep_prob = tf.placeholder(tf.float32) # probabilidade de drop_out no grafo

# augment dataset if training
tf_X = tf.cond(tf.less_equal(tf_keep_prob, 0.98), # usa prob de dropout para saber se está treinando
			   lambda: tf.map_fn(lambda image: augment(image), tf_X), # caso treinando, aumenta os dados
			   lambda: tf.map_fn(lambda image: center_crop(image), tf_X)) # caso testanto, corta imagen no centro


################ Variáveis do grafo ################
# Inicialização Xavier dos parâmetros
init_wc1 = np.sqrt(6.0 / (img_w * img_h * 1 + img_w * img_h * CL1_depth))
init_wc2 = np.sqrt(6.0 / (img_w * img_h * CL1_depth + img_w/2 * img_h/2 * CL2_depth))
init_wc3 = np.sqrt(6.0 / (img_w/2 * img_h/2 * CL2_depth + img_w/4 * img_h/4 * CL3_depth))
init_wfc1 = np.sqrt(6.0 / (img_w/4 * img_h/4 * CL3_depth + DL1_size))
init_wfc2 = np.sqrt(6.0 / (DL1_size + DL2_size))
init_out = np.sqrt(6.0 / (DL2_size + n_classes))

# Pesos da camada convolucional 1
wc1 = tf.Variable(tf.random_uniform([5, 5, 1, CL1_depth],
				minval=-init_wc1, maxval=init_wc1), name='wc1')

# Pesos da camada convolucional 2
wc2 = tf.Variable(tf.random_uniform([5, 5, CL1_depth, CL2_depth],
				minval=-init_wc2, maxval=init_wc2), name='wc2')

# Pesos da camada convolucional 3
wc3 = tf.Variable(tf.random_uniform([5, 5, CL2_depth, CL3_depth],
				minval=-init_wc3, maxval=init_wc3), name='wc3')

# Pesos da camada densa 1
wfc1 = tf.Variable(tf.random_uniform([img_h/4 * img_w/4 * CL3_depth, DL1_size],
				minval=-init_wfc1, maxval=init_wfc1), name='wfc1')

# Pesos da camada densa 2
wfc2 = tf.Variable(tf.random_uniform([DL1_size, DL2_size],
				minval=-init_wfc2, maxval=init_wfc2), name='wfc2')

# Pesos da camada de saida para cada dígito possível
wout_log1 = tf.Variable(tf.random_uniform([DL2_size, n_classes],
				minval=-init_out, maxval=init_out), name='wout_log1')

wout_log2 = tf.Variable(tf.random_uniform([DL2_size, n_classes],
				minval=-init_out, maxval=init_out), name='wout_log2')

wout_log3 = tf.Variable(tf.random_uniform([DL2_size, n_classes],
				minval=-init_out, maxval=init_out), name='wout_log3')

wout_log4= tf.Variable(tf.random_uniform([DL2_size, n_classes],
				minval=-init_out, maxval=init_out), name='wout_log4')

wout_log5= tf.Variable(tf.random_uniform([DL2_size, n_classes],
				minval=-init_out, maxval=init_out), name='wout_log5')

wout_log6= tf.Variable(tf.random_uniform([DL2_size, n_classes],
				minval=-init_out, maxval=init_out), name='wout_log6')

# Viés das camadas convolucionais
bc1 = tf.Variable(0.1 * tf.random_normal([CL1_depth]), name='bc1')
bc2 = tf.Variable(0.1 * tf.random_normal([CL2_depth]), name='bc2')
bc3 = tf.Variable(0.1 * tf.random_normal([CL3_depth]), name='bc3')

# Viés das camadas densas
bfc1 = tf.Variable(0.1 * tf.random_normal([DL1_size]), name='bfc1')
bfc2 = tf.Variable(0.1 * tf.random_normal([DL2_size]), name='bfc2')

# Viés das camadas de saida
bout_log1 = tf.Variable(0.1 * tf.random_normal([n_classes]), name='bout_log1')
bout_log2 = tf.Variable(0.1 * tf.random_normal([n_classes]), name='bout_log2')
bout_log3 = tf.Variable(0.1 * tf.random_normal([n_classes]), name='bout_log3')
bout_log4 = tf.Variable(0.1 * tf.random_normal([n_classes]), name='bout_log4')
bout_log5 = tf.Variable(0.1 * tf.random_normal([n_classes]), name='bout_log5')
bout_log6 = tf.Variable(0.1 * tf.random_normal([n_classes]), name='bout_log6')


################ Modelo de Rede Neural Convolucional ##################
# Primeira camada convolucional
conv1 = conv2d(tf_X, wc1, bc1, name='conv_l1')
conv1 = max_pool(conv1, ks=1, name='max_pool_l1') # max_pool sem down-sampling 

# Segunda camada convolucional
conv2 = conv2d(conv1, wc2, bc2, name='conv_l2')
conv2 = max_pool(conv2, name='max_pool_l2') # max_pool com down-sampling (diminui a imagem em um fator de 2)

# Terceira camada convolucional:
conv3 = conv2d(conv2, wc3, bc3, name='conv_l3')
conv3 = max_pool(conv3, name='max_pool_l3') # max_pool com down-sampling (diminui a imagem em um fator de 2)

# Primeira camada densa
fc1 = tf.reshape(conv3, [-1, wfc1.get_shape().as_list()[0]]) # Reformata o output de conv2 para passar à camada densa
fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, wfc1), bfc1), name='fc1') 
fc1 = tf.nn.dropout(fc1, tf_keep_prob, name='drop_out') # drop_outs

# Segunda camada densa
fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, wfc2), bfc2), name='fc2')

# Camada de saída
logits1 = tf.matmul(fc2, wout_log1) + bout_log1
logits2 = tf.matmul(fc2, wout_log2) + bout_log2
logits3 = tf.matmul(fc2, wout_log3) + bout_log3
logits4 = tf.matmul(fc2, wout_log4) + bout_log4
logits5 = tf.matmul(fc2, wout_log5) + bout_log5
logits6 = tf.matmul(fc2, wout_log6) + bout_log6
	

################ Computações de Treinamento ##################
# função custo com regularização
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits1, tf_y_input[:,0])) +\
	tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits2, tf_y_input[:,1])) +\
	tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits3, tf_y_input[:,2])) +\
	tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits4, tf_y_input[:,3])) +\
	tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits5, tf_y_input[:,4])) +\
	tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits6, tf_y_input[:,5])) +\
	gamma*tf.nn.l2_loss(wc1) +\
	gamma*tf.nn.l2_loss(wc2) +\
	gamma*tf.nn.l2_loss(wc3) +\
	gamma*tf.nn.l2_loss(wfc1) +\
	gamma*tf.nn.l2_loss(wfc2)

# otimizador
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

# Previsão
# Empilha a previsão de cada dígito para prever a imagem toda
prediction = tf.pack([logits1, logits2, logits3, logits4, logits5, logits6],
						axis=1, name='prediction')

saver = tf.train.Saver() # Para salvar o modelo
init = tf.global_variables_initializer() # para inicializar as variáveis



###################################################
############ Execução do Programa #################
###################################################

##### Confirma Uso Correto da Linha de Comando ####
# mensagem de erro
msg_1 = '''\n  Modo de usar: 'python decaptcha_cnn.py mode [imgs]'
	\r  em que 'mode' é algum dos dois abaixo:  
	\n\r 'train': quando estiver treinando com as imagens em train_imgs
	\n\r 'predict': quando estiver prevendo um CAPTCHA.
	\n\r   Nesse caso, deve-se passar uma lista '[imgs]'
	\r   com os arquivos dos CAPTCHAs\n'''

if len(run_flags) >= 1: # se tiver numero de args correto

	if run_flags[0] not in ['train', 'predict']: # checa args corretos
		print msg_1
		exit(1)

else: # se não tiver numero de args correto
	print msg_1
	exit(1)


################## Sessão TensorFlow ####################
with tf.Session() as sess:
	sess.run(init) # inicia as variáveis do grafo
	print '\n\n\n\n\n\n' # limpa o terminal dos outputs cuda

	
	# mode previsão
	if run_flags[0] == 'predict':
		
		ckpt = tf.train.get_checkpoint_state(save_dir) # acha o checkpoit

		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path) # restaura o modelo
			print "Sessão restaurada de ", ckpt.model_checkpoint_path, "."
		else:
			print 'Nenhum checkpoint encontrado' 
			exit(1)
		
		img_files = run_flags[1:] # pega os arquivos das imagens
		X_pred = get_test_data(img_files) # lê as imagens para teste

		# alimeta o modelo com as imagens
		feed_pred_dict = {tf_x_input : X_pred, tf_keep_prob:1.0}
		y_pred = sess.run(prediction, feed_dict=feed_pred_dict) # gera o vetor com scores

		for i in range(X_pred.shape[0]):

			cpt_figured = sample_digit(y_pred[i]) # sorteia os digitos comforme probabilidade
			plt.imshow(X_pred[i], interpolation='nearest', cmap='binary') # plota a imagem
			plt.title(cpt_figured, fontsize=40) # coloca o digito previsto como titulo
			plt.axis('off')
			plt.show()

			# espera verificação
			check = raw_input('Está correto?: [s/n]')
			
			# se estiver correto, renomeia o arquivo
			if check == 's':
				captcha_file = img_files[i]
				os.system("mv " + captcha_file + ' test_imgs/' + cpt_figured)

		exit(0)
	
	
	# modo treinamento
	if restore:
		ckpt = tf.train.get_checkpoint_state(save_dir)
		saver.restore(sess, ckpt.model_checkpoint_path)
		print "Sessão restaurada de ", ckpt.model_checkpoint_path, "."
	
	else:
		# apaga checkpoits antigos
		print 'Apagando checkpoits antigos'
		os.system("rm ./DECAPCHAcheckpoints/*") # comando linux para apagr arquivos


	X, y = get_data(5000) # carrega os dados

	# separa os dados em teste de treino e validação
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04, random_state=1)

	print 'Dimensões do set de treinamento: ', X_train.shape, y_train.shape
	print 'Dimensões do set de teste: ', X_test.shape, y_test.shape, '\n\n'
	
	valid_acc = [] # para parada do treinamento e checkpoints 
	train_loss = []
	t0 = time() # começa o cronometro
	no_inprove = 0

	
	# # para conferir vizualmente se o carregamento dos dados está correto
	# for i in range(3):
	# 	print '\n', np.argmax(y_train[i], axis=1)
	# 	plt.imshow(X_train[i], interpolation='nearest', cmap='binary')
	# 	plt.show()


	# loop de treinamento
	for step in range(training_iters):

		# faz os mini-lotes
		offset = (step * batch_size) % (y_train.shape[0] - batch_size)
		batch_data = X_train[offset:(offset + batch_size), :, :]
		batch_labels = y_train[offset:(offset + batch_size), :]
		
		if rotate:
			batch_data = rotate_batch(batch_data) # rotaciona o min-lote
		
		# roda uma iteração de treino
		feed_dict = {tf_x_input : batch_data, tf_y_input : batch_labels, tf_keep_prob: keep_prob_train}
		i_global, _, l = sess.run([global_step, optimizer, loss], feed_dict=feed_dict)
		train_loss.append(l)
		
		# mostra os resultados esporadicamente
		if (i_global % display_step == 0) or (step == training_iters - 1):
			print 'Tempo para rodar %d iterações:' % (step+1), round((time()-t0)/60, 3), 'min'
			
			feed_train_dict = {tf_x_input : batch_data, tf_keep_prob:1.0}
			train_pred = sess.run(prediction, feed_dict=feed_train_dict)
			acc_train = accuracy(train_pred, batch_labels) 
			print 'Custo no min-lote na iteração %d: %.4f' % (i_global, l)
			print 'Acurácia no min-lote: %.1f%%' % acc_train 
			
			feed_val_dict = {tf_x_input : X_test, tf_keep_prob:1.0}
			val_pred = sess.run(prediction, feed_dict=feed_val_dict)
			acc_val = accuracy(val_pred, y_test) 
			print 'Acurácia de validação: %.1f%% \n' % acc_val 
			
			# para parada do treinamento
			valid_acc.append(acc_val)
			
			# salva o modelo com parada adiantada 
			if acc_val >= np.max(valid_acc) and acc_val > min_val_acc:
				saver.save(sess, save_dir + 'model.ckpt', global_step=i_global+1)
				print("Checkpoint!\n")
				no_inprove = 0	
			
			# se 100 verificações se passaram sem avanço, sai do treinamento
			# no_inprove += 1
			# if no_inprove < 100:
			# 	break


	if show_learning:

		# pega uma grande amostra do set de treino (cuidado com o RAM)
		idx = np.random.randint(0, X_train.shape[0], 500)
		Xs_train =  X_train[idx, :, :]
		ys_train = y_train[idx, :]

		# passa a amostra pelo modelo
		feed_st_dict = {tf_x_input: Xs_train, tf_keep_prob:1.0}
		pred_st = sess.run(prediction, feed_dict = feed_st_dict)
		acc_st = accuracy(pred_st, ys_train)
		print 'Acurácia no set de treino: %.1f%%' % acc_st

		# plota a curva de custo por iteração
		plt.plot(range(len(train_loss)), train_loss)
		plt.show()