import cv2
import numpy as np

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25

id = input("Digite seu identificador: ")

largura, altura = 220, 220
print("Capturando as faces...")

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #print(np.average(imagemCinza))
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.5,
                                                     minSize=(100, 100))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
            
        if np.average(imagemCinza) > 110:
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite(f"fotos/pessoa.{id}.{amostra}.jpg", imagemFace)
                print(f"Foto {amostra} capturada com sucesso")
                amostra += 1
 
    cv2.imshow("Face", imagem)               
    
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

    #cv2.waitKey(1)

    if (amostra > numeroAmostras):
        break

print("Faces capturadas com sucesso")
camera.release()
cv2.destroyAllWindows()

# Alterações realizadas:
# 1. Adicionado import numpy as np
# 2. Adicionado classificador de olhos
# 3. Variável amostra adicionada
# 4. Variável numeroAmostras adicionada
# 5. Variável id adicionada
# 6. Variáveis largura e altura adicionadas
# 7. Adicionado print do valor médio da imagem em escala de cinza
# 8. Adicionada a variável regiao para a área do rosto
# 9. Adicionada a conversão da região para escala de cinza
# 10. Detectar olhos na região do rosto
# 11. Salvar a imagem do rosto
# 12. Condição para encerrar a captura após o número de amostras desejado
