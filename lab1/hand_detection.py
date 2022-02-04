import cv2
import mediapipe as mp

# Récupération des fonctions de dessin et de détéction
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Utilisation de la capture vidéo
cap = cv2.VideoCapture(0)

# Paramètres du model de détéction de main
with mp_hands.Hands(
        model_complexity=0,  # Entre 0 et 1, la complexité et la précision du modèle
        min_detection_confidence=0.5,  # Entre 0 et 1, seuil d'acceptation de la détéction de la main
        min_tracking_confidence=0.5  # Entre 0 et 1, seuil d'acceptation du tracking de la main
) as hands:
    # On procède au traitement tant que la capture caméra est active
    while cap.isOpened():
        success, image = cap.read()  # Récupération du flux de la caméra
        if not success:
            print("Echec de la récupération de l'image.")
            continue

        # On passe l'array de l'image en read_only et on la passe uniquement en traitement par référence pour améliorer
        # le traitement du modèle
        image.flags.writeable = False
        # On convertit l'image de BGR (OpenCV) à RGB (Mediapipe) pour le traitement
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # On stock le résultat du traitement du modèle
        results = hands.process(image)

        # On repasse l'image initiale en writeable pour pour dessiner dessus et on
        # la repasse en BGR afin qu'elle puisse être traitée par OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Si la détéction a aboutis à un résultat (varie selon les paramètres du modèle défini plus haut)
        if results.multi_hand_landmarks:
            # On dessine tous les marqueurs renvoyés par le modèle sur l'image
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # On retourne l'image pour un rendu type "Selfie"
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # On coupe le flux vidéo si l'utilisateur appuis sur "q"
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
cap.release()
