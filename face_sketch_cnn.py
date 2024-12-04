import os
from PIL import Image
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
from tensorflow.keras.models import load_model

def criar_modelo():
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(250, 200, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo

def listar_imagens(diretorio):
    try:
        lista_arquivos = os.listdir(diretorio)
        return lista_arquivos
    except FileNotFoundError:
        print(f"Erro: Diretório não encontrado {diretorio}")
        return []

def editar_imagens(diretorio_entrada, diretorio_saida, tamanho=(250, 200)):
    for nome_arquivo in os.listdir(diretorio_entrada):
        if nome_arquivo.endswith(".jpg"):
            imagem = Image.open(os.path.join(diretorio_entrada, nome_arquivo))
            imagem = imagem.resize(tamanho)
            array_imagem = np.array(imagem) / 255.0
            imagem_processada = Image.fromarray((array_imagem * 255).astype(np.uint8))
            caminho_imagem_processada = os.path.join(diretorio_saida, nome_arquivo)
            imagem_processada.save(caminho_imagem_processada)
            print(f"Processado: {nome_arquivo}")

def rotular_imagens(diretorio, arquivo_saida):
    with open(arquivo_saida, mode='w', newline='') as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(['nome_arquivo', 'rotulo'])
        for nome_arquivo in os.listdir(diretorio):
            if nome_arquivo.endswith(".jpg"):
                rotulo = 0 if nome_arquivo.startswith("m") else 1 if nome_arquivo.startswith("f") else None
                if rotulo is not None:
                    escritor.writerow([nome_arquivo, rotulo])
                    print(f"Arquivo: {nome_arquivo}, Rótulo: {rotulo}")

def carregar_dados(diretorio, arquivo_rotulos):
    imagens, rotulos = [], []
    with open(arquivo_rotulos, mode='r') as arquivo_csv:
        leitor = csv.reader(arquivo_csv)
        next(leitor)
        for linha in leitor:
            nome_arquivo, rotulo = linha
            caminho_imagem = os.path.join(diretorio, nome_arquivo)
            imagem = Image.open(caminho_imagem)
            array_imagem = np.array(imagem) / 255.0
            imagens.append(array_imagem)
            rotulos.append(int(rotulo))
    return np.array(imagens), np.array(rotulos)

diretorio_imagens_originais = "C:/Users/Alana França/Documents/imagens_originais"
diretorio_imagens_editadas = "C:/Users/Alana França/Documents/imagens_editadas"
caminho_arquivo_rotulos = "C:/Users/Alana França/Documents/arquivo_rotulos.csv"

imagens, rotulos = carregar_dados(diretorio_imagens_editadas, caminho_arquivo_rotulos)

X_treino, X_temporario, y_treino, y_temporario = train_test_split(imagens, rotulos, test_size=0.5, random_state=23)
X_validacao, X_teste, y_validacao, y_teste = train_test_split(X_temporario, y_temporario, test_size=0.4, random_state=23)

modelo = criar_modelo()
historico = modelo.fit(X_treino, y_treino, epochs=20, batch_size=32, validation_data=(X_validacao, y_validacao))

perda_teste, acuracia_teste = modelo.evaluate(X_teste, y_teste)
print(f'Acurácia no teste: {acuracia_teste * 100:.2f}%')

y_pred_probabilidades = modelo.predict(X_teste).ravel()
y_pred = (y_pred_probabilidades > 0.5).astype(int)

f1 = f1_score(y_teste, y_pred)
print(f'F1-Score: {f1:.2f}')

fpr, tpr, _ = roc_curve(y_teste, y_pred_probabilidades)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

indices_incorretos = np.where(y_teste != y_pred)[0]
print(f"Número de imagens incorretamente classificadas: {len(indices_incorretos)}")

for indice in indices_incorretos[:5]:
    img = X_teste[indice]
    plt.imshow(img)
    plt.title(f"Verdadeiro: {y_teste[indice]}, Predito: {y_pred[indice]}")
    plt.show()

plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Acurácia do modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.show()

plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Perda do modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.show()
