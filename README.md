# 🚗 Application Streamlit de comparaison automobile

Cette application Streamlit a été développée dans le cadre d’un projet de mise en œuvre d’un **Dashboard interactif** sur le **marché automobile**.  
Elle permet de comparer différents modèles de voitures à partir de données générées et enrichies, avec une **dimension de géolocalisation** pour contextualiser les analyses par région.

---

## 🧩 Fonctionnalités principales

1. **Analyse comparative des véhicules**  
   Filtrer, visualiser et comparer les modèles selon leurs caractéristiques (prix, puissance, consommation, kilométrage, carburant, etc.) et leur localisation géographique.

2. **Estimation de la probabilité d’achat**  
   Un modèle prédictif interne estime la probabilité qu’un prospect intéressé passe effectivement à l’achat, selon son profil et les attributs du véhicule.

3. **Analyse comportementale des clients**  
   Étude des différences entre **groupes d’âge**, de l’effet des **réductions commerciales**, et du **nombre de campagnes marketing** vues par prospect, afin d’identifier les facteurs qui influencent le plus la décision d’achat.

---

## 🎯 Objectif

Conçue pour un **usage interne en entreprise**, l’application aide les équipes marketing et commerciales à :
- identifier les caractéristiques qui influencent réellement le comportement d’achat ;
- mesurer l’efficacité des campagnes et des réductions ;
- comparer les performances des modèles de véhicules ;
- ajuster les stratégies selon les profils clients et les zones géographiques.

---

## ⚙️ Prérequis

- **Docker**
- **Python 3.12**

> Aucune autre configuration n’est nécessaire. Le script charge automatiquement les variables d’environnement depuis le fichier `.env` inclus dans le projet (non versionné).

---

## ▶️ Étapes pour lancer l’application

1. **Cloner le dépôt**
   ```bash
   git clone <URL_DU_DEPOT>
   
2. **Se placer dans le dossier du projet**
   ```bash
   cd projet_streamlit_linux
3. **Lancer le script de déploiement**
   ```bash
     bash deploy.sh

Le script :

construit automatiquement l’image Docker,

exécute le conteneur Streamlit,

charge les variables d’environnement depuis .env,

et démarre l’application

## 🌐 Accès à l’application
Une fois le script lancé, ouvrez votre navigateur à l’adresse :
http://localhost:8501

Tout est automatisé : aucune installation manuelle ni configuration supplémentaire.
Pour arrêter le conteneur, utilisez Ctrl + C dans le terminal.


