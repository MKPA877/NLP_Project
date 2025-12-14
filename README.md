# AKAlearn Quiz - Générateur d'Examens par IA

AKAlearn Quiz est une application interactive développée avec **Streamlit** qui utilise l'intelligence artificielle (via l'API **Groq** et le modèle **Llama 3**) pour générer automatiquement des quiz et des examens à partir de n'importe quel texte source.

## 🚀 Fonctionnalités

- **Génération de questions multi-formats** :
  - ✅ **Vrai/Faux** : Pour tester la compréhension rapide.
  - 📝 **QCM** : Questions à choix multiples avec 4 options.
  - ✍️ **Questions Ouvertes** : Réponses courtes nécessitant une rédaction.
- **Niveaux de difficulté adaptatifs** : Facile, Moyen, Difficile, Expert.
- **Correction Intelligente** :
  - Correction automatique instantanée pour les QCM et Vrai/Faux.
  - **Correction par IA** pour les questions ouvertes : analyse sémantique de la réponse, attribution d'un score et feedback constructif.
- **Interface personnalisable** : Ajustement de la "créativité" du modèle (température) et du nombre de questions.

## 🛠️ Prérequis

- Python 3.8 ou supérieur
- Une clé API **Groq** (gratuite en version bêta) : [Obtenir une clé ici](https://console.groq.com/)

## 📦 Installation

1. **Cloner le projet** (ou télécharger les fichiers) :
   ```bash
   git clone <votre-repo>
   cd <dossier-du-projet>
   ```

2. **Installer les dépendances** :
   ```bash
   pip install streamlit groq python-dotenv pydantic
   ```

3. **Configuration de l'environnement** :
   Créez un fichier `.env` à la racine du projet et ajoutez votre clé API Groq :
   ```env
   GROQ_API_KEY=gsk_votre_cle_api_ici...
   ```

## ▶️ Utilisation

1. **Lancer l'application** :
   ```bash
   streamlit run app2.py
   ```

2. **Dans votre navigateur** :
   - Collez un texte (cours, article, résumé) dans la barre latérale.
   - Choisissez le type de question, la difficulté et le nombre de questions.
   - Cliquez sur **"Générer l'exam"**.
   - Répondez aux questions et cliquez sur **"Vérifier réponses et correction"** pour voir votre score et les explications.

## 🏗️ Architecture Technique

- **Frontend** : Streamlit
- **Backend IA** : Groq SDK (Modèle `llama-3.1-8b-instant`)
- **Validation de données** : Pydantic (Assure que l'IA génère un format JSON strict et exploitable).

## 📝 Structure du Projet

- `app2.py` : Code principal de l'application.
- `.env` : Fichier de configuration pour les clés API (à ne pas partager).
