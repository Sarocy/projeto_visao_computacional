import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        print(id)
        ids.append(id)
        faces.append(imagemFace)
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)
    return np.array(ids), faces


ids, faces = getImagemId()

print("treinando...")
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado")


def calcular_precisao(previstos, verdadeiros):
    total = len(previstos)
    corretos = sum(1 for p, v in zip(previstos, verdadeiros) if p == v)
    return corretos / total if total > 0 else 0


def treinar_modelo():
    previstos = [0, 1, 0, 1, 1]  # Resultados previstos
    verdadeiros = [0, 1, 0, 0, 1]  # Rótulos verdadeiros

    precisao = calcular_precisao(previstos, verdadeiros)
    print(f'Precisão: {precisao * 100:.2f}%')


if __name__ == "__main__":
    treinar_modelo()

    import matplotlib.pyplot as plt
    import numpy as np


    def treinar_modelo():
        epocas = 10
        precisoes = np.random.rand(epocas)  # Exemplo de precisão aleatória
        perdas = np.random.rand(epocas)  # Exemplo de perda aleatória

        plt.figure(figsize=(10, 5))

        # Plotar precisão
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epocas + 1), precisoes, marker='o', linestyle='-', color='b', label='Precisão')
        plt.xlabel('Épocas')
        plt.ylabel('Precisão')
        plt.title('Precisão do Modelo ao Longo do Treinamento')
        plt.xticks(range(1, epocas + 1))
        plt.legend()
        plt.grid(True)

        # Plotar perda
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epocas + 1), perdas, marker='o', linestyle='-', color='r', label='Perda')
        plt.xlabel('Épocas')
        plt.ylabel('Perda')
        plt.title('Perda do Modelo ao Longo do Treinamento')
        plt.xticks(range(1, epocas + 1))
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    if __name__ == "__main__":
        treinar_modelo()

        import cv2
        import numpy as np
        import os
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        import matplotlib.pyplot as plt
        import seaborn as sns


        def carregar_imagens_e_identificadores(diretorio):
            imagens = []
            identificadores = []
            for root, dirs, files in os.walk(diretorio):
                for file in files:
                    if file.endswith('.jpg'):
                        path = os.path.join(root, file)
                        identificador = int(os.path.basename(path).split('_')[0])
                        imagem = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        imagens.append(imagem)
                        identificadores.append(identificador)
            return imagens, np.array(identificadores)


        if __name__ == "__main__":
            diretorio_teste = 'caminho/para/seu/diretorio_de_teste'
            imagens_teste, identificadores_verdadeiros = carregar_imagens_e_identificadores(diretorio_teste)

            recognizer = cv2.face.EigenFaceRecognizer_create()

            caminho_modelo = 'C:/caminho/para/seu/modelo_treinado.yml'

            if os.path.exists(caminho_modelo):
                recognizer.read(caminho_modelo)
            else:
                print(f"Erro: Arquivo '{caminho_modelo}' não encontrado.")
                exit()

            previsoes = []

            # Testar o modelo com imagens de teste
            for imagem in imagens_teste:
                label, confidence = recognizer.predict(imagem)
                previsoes.append(label)

            matriz_confusao = confusion_matrix(identificadores_verdadeiros, previsoes)

            accuracy = accuracy_score(identificadores_verdadeiros, previsoes)
            precision = precision_score(identificadores_verdadeiros, previsoes, average='weighted')
            recall = recall_score(identificadores_verdadeiros, previsoes, average='weighted')
            f1 = f1_score(identificadores_verdadeiros, previsoes, average='weighted')


            print(f'Acurácia: {accuracy:.2f}')
            print(f'Precisão: {precision:.2f}')
            print(f'Recall: {recall:.2f}')
            print(f'F1-score: {f1:.2f}')

            # Visualizar a matriz de confusão usando seaborn
            plt.figure(figsize=(8, 6))
            sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Previsão')
            plt.ylabel('Verdadeiro')
            plt.title('Matriz de Confusão')
            plt.show()

