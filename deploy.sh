#!/bin/bash

set -e 

echo "Lancement du déploiement de l'application"
echo 


#Création d'un environnement virtuel s'il n'y en n'a pas

if [[ ! -d ".venv" ]]; then  
echo "Création de l'environnement virtuel"
python3 -m venv .venv
fi
echo "L'environnement virtuel a été crée, passons à l'activation" 
echo  



#Activation de l'environnement virtuel 

echo "Activation de l'environnement virtuel"
source .venv/bin/activate
echo "L'environnement virtuel est activé"
echo



#Vérifier si poetry est installé, sinon l'installer

if ! command -v poetry ; then 
echo "Poetry n'est pas installé. Installation en cours "
python -m pip install --upgrade pip
python -m pip install poetry 
echo " Poetry a été installé"

else 
echo "Poetry est déjà installé, passons à l'installation des dépendances"
fi 
echo


#Installation des dépendances 
echo "Installation des dépendances"
poetry install --no-root --no-interaction --no-ansi 

echo "Les dépendances ont été installées"
echo

# Lancement de l'application 

PORT="${PORT:-8501}"
echo " Lancement de l'application streamlit"
docker run -p "${PORT}:${PORT}" streamlit_cars






  
