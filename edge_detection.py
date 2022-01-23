import cv2

def main():
    
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    while True : 
        sucess, img = cap.read()#recup flux cam
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#passe l'image en niveau de gris
        img_blur = cv2.GaussianBlur(img_gray, (13,13),0) #floute l'image pour enlever du bruit
        edges = cv2.Canny(img_blur, 0, 30, 80) #Recupere les bords
        cv2.imshow('edges image', edges) #permet de voir l'image generer pour trouver des meilleurs threshold
    
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #recupère les contours
        cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)#met les contours sur l'image original
        cv2.imshow('Result', img)#affiche image 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()