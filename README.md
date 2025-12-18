# Application Streamlit de comparaison automobile

Cette application Streamlit a été développée dans le cadre d’un projet de mise en œuvre d’un **Dashboard interactif** sur le **marché automobile**.  
Elle permet de comparer différents modèles de voitures à partir de données générées et enrichies, avec une **dimension de géolocalisation** pour contextualiser les analyses par région.

---

## I - Fonctionnalités principales

1. **Analyse comparative des véhicules**  
   Filtrer, visualiser et comparer les modèles selon leurs caractéristiques (prix, puissance, consommation, kilométrage, carburant, etc.) et leur localisation géographique.

2. **Estimation de la probabilité d’achat**  
   Un modèle prédictif interne estime la probabilité qu’un prospect intéressé passe effectivement à l’achat, selon son profil et les attributs du véhicule.

3. **Analyse comportementale des clients**  
   Étude des différences entre **groupes d’âge**, de l’effet des **réductions commerciales**, et du **nombre de campagnes marketing** vues par prospect, afin d’identifier les facteurs qui influencent le plus la décision d’achat.

---

## II - Objectif

Conçue pour un **usage interne en entreprise**, l’application aide les équipes marketing et commerciales à :
- identifier les caractéristiques qui influencent réellement le comportement d’achat ;
- mesurer l’efficacité des campagnes et des réductions ;
- comparer les performances des modèles de véhicules ;
- ajuster les stratégies selon les profils clients et les zones géographiques.

---

## ⚙️ Prérequis

- **Docker**

> Aucune autre configuration n’est nécessaire. Le script charge automatiquement les variables d’environnement depuis le fichier `.env` inclus dans le projet (non versionné).
**Important** : N'oubliez pas de **lancer l'application Docker Desktop** avant de passer aux étapes suivantes. Le script de déploiement échouera si Docker n'est pas actif.

---

## ▶️ Étapes pour lancer l’application

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/Marwan29o/projet_streamlit_linux.git
   
2. **Se placer dans le dossier du projet**
   ```bash
   cd projet_streamlit_linux

3. **Télécharger les données depuis Google Drive**
   ```bash
     bash data_collector/bin/get_data.sh

3. **Lancer le script de déploiement**
   ```bash
     bash deploy.sh

Le script :

• construit automatiquement l’image Docker,

• exécute le conteneur Streamlit,

• prépare l'environnement,

• et démarre l’application

## III - Accès à l’application
Une fois le script lancé, ouvrez votre navigateur à l’adresse :
http://localhost:8501

Tout est automatisé : aucune installation manuelle ni configuration supplémentaire.
Pour arrêter le conteneur, utilisez Ctrl + C dans le terminal.

### 🛑 Arrêter un conteneur Docker encore en cours d’exécution

Si vous ne stoppez pas manuellement le conteneur Docker, **il continue de tourner en arrière-plan**, même après avoir fermé le terminal.  
Cela peut bloquer le port 8501 et empêcher de relancer l’application.

Voici comment le stopper proprement :

1. **Lister les conteneurs actifs :**
   ```bash
   docker ps

2. **Repérer l’ID du conteneur lié à l’application :**
(colonne **CONTAINER ID**, généralement associé à l’image streamlit_cars).

3. **Stopper le conteneur :**
   ```bash
   docker stop <ID_DU_CONTENEUR>

### Accéder à l’application

Une fois le script exécuté et l’application lancée, ouvre ton navigateur et va sur :

**http://localhost:8501**

**Ne pas utiliser http://0.0.0.0:8501.**
Cette adresse n’est affichée que dans le terminal pour indiquer que le serveur écoute sur toutes les interfaces, mais elle ne fonctionne pas dans un navigateur.




