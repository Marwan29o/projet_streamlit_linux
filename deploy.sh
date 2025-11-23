#!/bin/bash
set -e

echo "=== Déploiement de l'application Streamlit ==="

# 1) Vérifier Docker
if ! command -v docker >/dev/null 2>&1; then
    echo "ERREUR : Docker n'est pas installé sur cette machine."
    echo "Veuillez installer Docker Desktop avant de lancer le script."
    exit 1
fi

# 2) Build Docker
echo "Construction de l'image Docker..."
docker build -t streamlit_cars .

echo "Image Docker construite avec succès."
echo

# 3) Lancement
PORT="${PORT:-8501}"

echo "Lancement de l'application sur http://localhost:$PORT"
docker run -p "${PORT}:${PORT}" streamlit_cars