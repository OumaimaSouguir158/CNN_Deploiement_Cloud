# 🔴 Projet 4 — Classification d'images par Deep Learning (CNN) et déploiement Cloud

## Objectif
Entraîner un réseau de neurones convolutif (par transfer learning) puis le déployer sur une plateforme Cloud accessible via une API.

## Fonctionnalités
- Entraînement/fine-tuning d'un CNN (MobileNetV2 pré-entraîné sur ImageNet + tête de classification sur CIFAR-10)
- Hébergement du modèle sur une plateforme Cloud (offre gratuite)
- API d'inférence exposée publiquement (upload d'image → classe prédite)

## Stack technique
Python · TensorFlow/Keras · service Cloud (Render / Google Cloud Run / Hugging Face Spaces) · Docker

## Concepts démontrés
Deep learning, transfer learning, déploiement Cloud, scalabilité d'un service d'inférence.

## Structure du projet
```
04_CNN_Deploiement_Cloud/
├── training/
│   └── train_cnn.py         # entraînement par transfer learning (MobileNetV2)
├── app/
│   ├── main.py               # API FastAPI d'inférence
│   ├── requirements.txt
│   └── model/                # SavedModel généré par train_cnn.py
└── Dockerfile
```

## Comment lancer le projet
1. **Entraîner le modèle :**
   ```bash
   cd training && pip install tensorflow && python train_cnn.py
   ```
2. **Tester l'API en local :**
   ```bash
   cd app && pip install -r requirements.txt
   uvicorn main:app --reload
   curl -X POST -F "file=@photo_chat.jpg" http://localhost:8000/predict
   ```
3. **Déploiement Cloud (exemple Google Cloud Run) :**
   ```bash
   gcloud builds submit --tag gcr.io/PROJET/cnn-api
   gcloud run deploy cnn-api --image gcr.io/PROJET/cnn-api --platform managed --allow-unauthenticated
   ```

## Ligne CV
« CNN + déploiement Cloud — transfer learning, API d'inférence hébergée. »

## Question d'entretien possible
Quels sont les défis de scalabilité lorsqu'un service d'inférence Cloud reçoit un pic de requêtes ?

*(Réponse : mise à l'échelle horizontale automatique (autoscaling) du nombre d'instances, gestion du cold start des conteneurs, mise en cache des résultats fréquents, limitation du temps de réponse en dégradant la charge (batching des requêtes), et surveillance des coûts liés à l'auto-scaling.)*
