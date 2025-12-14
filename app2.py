import streamlit as st
import os
import json
import time
import random
import re 
from dotenv import load_dotenv 
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union
from groq import Groq 
from json.decoder import JSONDecodeError
import hashlib

#Structures Pydantic pour génération et correction

class VraiFauxQuestion(BaseModel):
    question: str = Field(description="Le corps de la question, qui doit être une affirmation.")
    correct_answer: bool = Field(description="La réponse correcte: True si l'affirmation est Vraie (True), False sinon (False).")
    topic: str = Field(description="Le concept testé (courte phrase, 2-6 mots).")

class MCQQuestion(BaseModel):
    question: str = Field(description="Le corps de la question.")
    options: List[str] = Field(description="Une liste de 4 options de réponse, incluant la bonne.")
    correct_answer: str = Field(description="Le texte exact de la bonne réponse.")
    topic: str = Field(description="Le concept testé (courte phrase, 2-6 mots).")

class QuestionOuverte(BaseModel):
    question: str = Field(description="La question ouverte nécessitant une réponse courte.")
    keywords: List[str] = Field(description="Une liste de 3-5 mots-clés ou phrases courtes attendus dans la réponse de l'utilisateur.")
    topic: str = Field(description="Le concept testé (courte phrase, 2-6 mots).")

class CorrectionFeedback(BaseModel):
    """Structure pour le retour de correction du LLM pour les questions ouvertes."""
    score_percentage: int = Field(description="Pourcentage de justesse de la réponse de l'utilisateur (0 à 100).")
    feedback_text: str = Field(description="Une explication courte et constructive justifiant le score et indiquant ce qui manque ou ce qui est correct.")
    is_correct: bool = Field(description="True si le score_percentage est >= 70, False sinon.")

#Config Groq

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    try:
        LLM_CLIENT = Groq(api_key=GROQ_API_KEY)
        LLM_MODEL = "llama-3.1-8b-instant"  
    except Exception as e:
        LLM_CLIENT = None
        st.error(f"ERREUR d'initialisation du client Groq : {e}")
else:
    LLM_CLIENT = None
    st.error(
        "ERREUR: La clé 'GROQ_API_KEY' est manquante.\n"
        "Veuillez définir la variable d'environnement GROQ_API_KEY dans le fichier .env."
    )

#Config difficulte
def get_difficulty_instructions(difficulty: str) -> str:
    """Fournit des instructions spécifiques au LLM pour moduler la complexité cognitive."""
    if difficulty == "Facile":
        return "Les questions doivent se concentrer sur les définitions et les faits directement cités, ne nécessitant aucune inférence."
    elif difficulty == "Moyen":
        return "Les questions doivent porter sur la compréhension des relations de cause à effet et l'application simple des concepts décrits."
    elif difficulty == "Difficile":
        return "Les questions doivent exiger la synthèse de plusieurs idées, la justification d'un processus, ou l'interprétation de données non explicites."
    elif difficulty == "Expert":
        return "Les questions doivent nécessiter une analyse critique approfondie, une comparaison d'éléments ou une extrapolation des conséquences au-delà du texte immédiat."
    return ""
#Fonctions génération et validation ---

def get_target_config(type_question: str):
    if type_question == "Vrai/Faux":
        return VraiFauxQuestion, "VRAI/FAUX : chaque question doit être une affirmation à laquelle on répond par True ou False.", [
            {"question": "L'eau bout à 100 degrés Celsius au niveau de la mer.", "correct_answer": True, "topic": "Point d'ébullition"}
        ]
    elif type_question == "Ouvert":
        return QuestionOuverte, "OUVERTE : chaque question doit nécessiter une réponse courte. Le champ 'keywords' doit contenir les éléments de correction.", [
            {"question": "Quelle est la formule chimique de l'eau?", "keywords": ["H2O", "hydrogène", "oxygène"], "topic": "Formule chimique"}
        ]
    else: 
        return MCQQuestion, "QCM : chaque question doit avoir 4 options (options), dont une seule est la bonne (correct_answer).", [
            {"question": "Quel gaz est produit à la cathode lors de l'électrolyse de l'eau?", "options": ["Oxygène", "Hydrogène", "Azote", "Méthane"], "correct_answer": "Hydrogène", "topic": "Électrolyse de l'eau"}
        ]

def extract_and_validate_json(json_string: str, target_model: BaseModel) -> List[BaseModel]:
    #Nettoie le texte brut du LLM et valide le JSON avec Pydantic.
    try:
        # Tente d'isoler le tableau JSON ([...]).
        start_index = json_string.find('[')
        end_index = json_string.rfind(']')
        
        # Gère les cas où le LLM produit un objet ({...}) au lieu d'un tableau ou une structure non conforme.
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            if json_string.strip().startswith('{') and target_model != CorrectionFeedback:
                 try:
                    data = json.loads(json_string)
                    if isinstance(data, dict):
                         data = [data] 
                    else:
                         raise ValueError
                 except (json.JSONDecodeError, ValueError):
                      raise ValueError("Délimiteurs JSON ([...] ou {...}) non trouvés ou structure non conforme.")
            # Cas spécifique pour la correction (un seul objet de feedback attendu).
            elif target_model == CorrectionFeedback:
                data = json.loads(json_string)
                if not isinstance(data, dict):
                    raise ValueError("Sortie JSON invalide : Le contenu n'est pas un objet (dict).")
                valid_feedback = target_model(**data)
                return [valid_feedback] 
            else:
                 raise ValueError("Délimiteurs de tableau JSON ([...]) non trouvés.")
        else:
            # Coupe le texte pour ne garder que le tableau JSON propre.
            json_to_parse = json_string[start_index : end_index + 1]
            data = json.loads(json_to_parse)

    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Erreur de décodage JSON : {e}. Sortie LLM brute non conforme au format JSON.")
    
    if not isinstance(data, list):
        if isinstance(data, dict):
             data = next((v for k, v in data.items() if isinstance(v, list)), data)
        
        if not isinstance(data, list):
             raise ValueError("Sortie JSON invalide : Le contenu n'est pas un tableau (list) principal.")
    
    valid_questions = []
    
    # Valide élément de la liste avec le modèle Pydantic attendu.
    for item in data:
        if not isinstance(item, dict):
             continue
        try:
            valid_questions.append(target_model(**item))
        except ValidationError:
             continue

    if not valid_questions:
        raise ValueError("Aucun mapping valide trouvé après le filtrage Pydantic.")
        
    return valid_questions

def generate_questions(text_source: str, difficulty: str, num_questions: int, type_question: str, temperature: float) -> List[BaseModel]:
    if not LLM_CLIENT:
        return []
    
    target_model, schema_description, example_output = get_target_config(type_question)
    target_schema = target_model.model_json_schema()

    example_output_str = json.dumps(example_output, indent=2)
    
    difficulty_instruction = get_difficulty_instructions(difficulty)

    system_prompt = f"""
    Vous êtes un assistant pédagogique expert générant des questions de type {type_question} ({schema_description}) basées sur le texte fourni.
    Le niveau de difficulté doit être strictement : {difficulty}. {difficulty_instruction}
    
    INSTRUCTION CRITIQUE: Votre sortie DOIT être STRICTEMENT un tableau JSON (JSON array) contenant exactement {num_questions} objets, SANS AUCUN AUTRE TEXTE NI MARKDOWN avant ou après.
    
    EXEMPLE DE STRUCTURE ATTENDUE (N'UTILISEZ PAS CE CONTENU, SEULEMENT LE FORMAT):
    {example_output_str}
    
    SCHEMA JSON (POUR RÉFÉRENCE) :
    {json.dumps(target_schema, indent=2)}
    """
    
    MAX_RETRIES = 3
    #Retry pour tolérer les erreurs de formatage JSON du LLM.
    for attempt in range(MAX_RETRIES):
        try:
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Générer {num_questions} questions basées sur le texte suivant : \n\n{text_source}"}
                ],
                response_format={"type": "json_object"}, 
                temperature=temperature 
            )
            
            json_string = response.choices[0].message.content.strip()
            
            questions = extract_and_validate_json(json_string, target_model)
            
            if len(questions) < num_questions:
                 st.warning(f"Attention : Le modèle a généré seulement {len(questions)} questions valides au lieu de {num_questions} demandées.")

            return questions

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                st.warning(f"Erreur LLM/Pydantic (Tentative n°{attempt+1} échouée : {e}). Nouvelle tentative dans {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error(f"Échec après {MAX_RETRIES} tentatives pour cause d'erreur LLM : {e}")
                st.error("Échec critique de la génération JSON structurée. Simplifiez le texte source, augmentez la température, ou vérifiez la clé API.")
                return []
    return []

#Correction par LLM

def get_llm_feedback(question_text: str, user_answer: str, expected_keywords: List[str]) -> CorrectionFeedback:
    """Demande au LLM de corriger une question ouverte et de donner un feedback structuré."""
    # Fonction dédiée à la correction des questions ouvertes
    if not LLM_CLIENT:
        return CorrectionFeedback(score_percentage=0, feedback_text="Erreur de configuration LLM.", is_correct=False)
    
    target_schema = CorrectionFeedback.model_json_schema()
    
    system_prompt = f"""
    Vous êtes un correcteur pédagogique expert. Votre tâche est d'évaluer la justesse d'une réponse utilisateur à une question ouverte.
    
    Question : {question_text}
    Mots-clés attendus (pour référence) : {', '.join(expected_keywords)}
    Réponse de l'utilisateur : {user_answer}
    
    INSTRUCTION CRITIQUE: Votre sortie DOIT être STRICTEMENT un OBJET JSON, SANS AUCUN AUTRE TEXTE NI MARKDOWN avant ou après.
    Évaluez la réponse et attribuez un score en pourcentage (0-100). Définissez 'is_correct' à True si le score est >= 70.
    
    SCHEMA JSON (POUR SORTIE) :
    {json.dumps(target_schema, indent=2)}
    """
    
    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES):
        try:
            response = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            json_string = response.choices[0].message.content.strip()
            
            # Validation correction
            feedback_list = extract_and_validate_json(json_string, CorrectionFeedback)
            
            return feedback_list[0] if feedback_list else CorrectionFeedback(score_percentage=0, feedback_text="LLM n'a pas pu structurer le feedback (erreur interne ou format invalide).", is_correct=False)

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1) 
            else:
                st.error(f"Échec après {MAX_RETRIES} tentatives de correction LLM : {e}")
                return CorrectionFeedback(score_percentage=0, feedback_text=f"Échec critique de la correction LLM : {e}", is_correct=False)
    
    return CorrectionFeedback(score_percentage=0, feedback_text="Échec inconnu de la correction LLM.", is_correct=False)

#Affichage test

def reset_all_data(full_reset=False):
    """Nettoie l'état de session lié aux questions, réponses et résultats."""
    st.session_state['show_results'] = False
    
    # Supprime toutes les clés de session liées aux réponses et corrections pour repartir à zéro.
    keys_to_delete = [
        k for k in list(st.session_state.keys()) 
        if k.startswith('q_') or k.startswith('feedback_q_') or k.endswith('_radio') or k.endswith('_text')
    ]
    for k in keys_to_delete:
        if k in st.session_state:
            del st.session_state[k]
    
    if full_reset:
        st.session_state['questions_data'] = None 
        st.success("Examen réinitialisé. Veuillez générer de nouvelles questions.")

def reset_show_results():
    """Callback pour désactiver l'affichage des résultats et forcer la regénération du feedback."""
    # Efface les anciens feedbacks pour qu'ils soient recalculés si l'utilisateur change une réponse.
    st.session_state['show_results'] = False
    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith('feedback_q_')]
    for k in keys_to_delete:
        if k in st.session_state:
            del st.session_state[k]

def update_user_answer(widget_key: str, q_key: str):
    """Callback: met à jour la réponse de l'utilisateur et désactive l'affichage des résultats."""
    st.session_state[q_key] = st.session_state.get(widget_key)
    reset_show_results()
    
def display_question_test(question, index: int, show_results: bool, results: dict = None):
    """Affiche une question pour le test et enregistre la réponse de l'utilisateur."""

    q_type = question.get('__type__')
    qid = question.get('id')
    q_key = f"q_{qid}_answer"
    q_text = question.get('question')
    q_topic = question.get('topic')
    
    PLACEHOLDER_RADIO = "Choisir la bonne réponse"

    col_q, col_status = st.columns([0.9, 0.1])
    
    with col_q:
        st.markdown(f"#### Q{index+1}. {q_text}")
    
    if show_results and results:
        status = results.get('is_correct')
        user_answered = (results.get('user_text') != "Non répondu")
        
        with col_status:
            if status:
                st.success("✅") 
            elif user_answered:
                st.error("❌")
            else:
                st.warning("❓")
        
    st.caption(f"Type: {q_type} | Thème: {q_topic}")

    # Logique d'affichage des widgets spécifiques (radio pour QCM/VraiFaux, text_area pour Ouvert)
    if q_type == 'MCQQuestion':
        options_key = f"q_{qid}_options"
        if options_key not in st.session_state:
            opts = question.get('options')[:]
            random.shuffle(opts)
            st.session_state[options_key] = opts

        options_mixed = st.session_state[options_key]
        displayed_options = [PLACEHOLDER_RADIO] + options_mixed

        radio_widget_key = f"q_{qid}_radio"
        st.session_state.setdefault(radio_widget_key, st.session_state.get(q_key, PLACEHOLDER_RADIO))
        if st.session_state[radio_widget_key] not in displayed_options:
            st.session_state[radio_widget_key] = PLACEHOLDER_RADIO
        
        st.radio(
            label="Choisissez une option :",
            options=displayed_options,
            key=radio_widget_key,
            on_change=update_user_answer,
            args=(radio_widget_key, q_key)
        )

    elif q_type == 'VraiFauxQuestion':
        # Logique similaire au QCM, mais avec seulement Vrai/Faux.
        options = ["Vrai", "Faux"]
        displayed_options = [PLACEHOLDER_RADIO] + options

        radio_widget_key = f"q_{qid}_radio"
        st.session_state.setdefault(radio_widget_key, st.session_state.get(q_key, PLACEHOLDER_RADIO))

        if st.session_state[radio_widget_key] not in displayed_options:
            st.session_state[radio_widget_key] = PLACEHOLDER_RADIO
            
        st.radio(
            label="Est-ce une affirmation Vraie ou Fausse :",
            options=displayed_options,
            key=radio_widget_key,
            on_change=update_user_answer,
            args=(radio_widget_key, q_key)
        )

    elif q_type == 'QuestionOuverte':
        # Widget text_area pour la réponse libre.
        text_widget_key = f"q_{qid}_text"
        st.session_state.setdefault(text_widget_key, st.session_state.get(q_key, ""))

        st.text_area(
            label="Votre réponse :",
            key=text_widget_key,
            value=st.session_state.get(text_widget_key, ""), 
            on_change=update_user_answer,
            args=(text_widget_key, q_key)
        )

#Logique correction

def check_results():
    """Déclenche l'affichage des résultats."""
    st.session_state['show_results'] = True

def calculate_score(questions_list):
    """Calcule le score et retourne les détails de la correction."""
    score = 0
    total = len(questions_list)
    results = []
    
    PLACEHOLDER_RADIO = "Choisir la bonne réponse"
    EMPTY_RESPONSE = "Non répondu"

    # Vérifie si une correction LLM est nécessaire pour les questions ouvertes.
    is_open_question_correction_needed = any(
        q.get('__type__') == 'QuestionOuverte' and f"feedback_q_{q.get('id')}" not in st.session_state
        for q in questions_list
    )
    
    if is_open_question_correction_needed:
        st.info("Correction des questions ouvertes en cours par Llama 3 (via Groq)...")
        with st.empty():
            pass

    for i, question in enumerate(questions_list):
        qid = question.get('id')
        q_key = f"q_{qid}_answer"
        user_choice = st.session_state.get(q_key)
        is_correct = False
        user_text = EMPTY_RESPONSE
        llm_feedback = None 

        if user_choice is None or user_choice == PLACEHOLDER_RADIO or (isinstance(user_choice, str) and not user_choice.strip()):
            user_choice = None
        
        correct_text = ""

        # Logique de correction pour les questions QCM, Vrai/Faux
        if question.get('__type__') == 'MCQQuestion':
            correct_text = question.get('correct_answer')
            if user_choice:
                user_text = user_choice
                is_correct = (user_choice == correct_text)
            
        elif question.get('__type__') == 'VraiFauxQuestion':
            correct_answer_val = question.get('correct_answer')
            correct_answer_str = "Vrai" if correct_answer_val else "Faux"
            correct_text = correct_answer_str
            if user_choice:
                user_text = user_choice
                is_correct = (user_choice == correct_text)
            
        # Logique de correction pour les questions ouvertes (avec appel au LLM si nécessaire)
        elif question.get('__type__') == 'QuestionOuverte':
            keywords = question.get('keywords')
            correct_text = f"Mots-clés attendus : {', '.join(keywords)}"
            
            if user_choice and user_choice.strip():
                user_text = user_choice
                
                feedback_key = f"feedback_q_{qid}"
                
                # Correction LLM (uniquement si les résultats sont affichés ET que le feedback n'a pas encore été calculé)
                if feedback_key not in st.session_state and st.session_state['show_results']:
                    feedback_obj = get_llm_feedback(
                        question_text=question.get('question'),
                        user_answer=user_choice,
                        expected_keywords=keywords
                    )
                    llm_feedback = feedback_obj.model_dump()
                    st.session_state[feedback_key] = llm_feedback
                
                llm_feedback = st.session_state.get(feedback_key)

                if llm_feedback:
                    is_correct = llm_feedback['is_correct']
            
        if is_correct:
            score += 1
        
        results.append({
            'index': i + 1, 
            'question': question.get('question'),
            'topic': question.get('topic'),
            'user_text': user_text,
            'correct_text': correct_text,
            'is_correct': is_correct,
            'llm_feedback': llm_feedback 
        })

    return score, total, results


#Streamlit

def main():
    st.set_page_config(page_title="Générateur d'exams LLM", layout="wide")
    st.title("AKAlearn Quiz")
    st.markdown("---")
    
    # Initialiser état de session
    if 'questions_data' not in st.session_state:
         st.session_state['questions_data'] = None 
    if 'show_results' not in st.session_state:
         st.session_state['show_results'] = False 


    with st.sidebar:
        st.header("Paramètres de l'exam")
        
        text_source = st.text_area(
            "Collez le texte ici :", 
            value="",
            height=200
        )
        
        difficulty = st.selectbox("Difficulté :", options=["Facile", "Moyen", "Difficile", "Expert"])
        type_question = st.selectbox("Type de Question :", options=["QCM", "Vrai/Faux", "Ouvert"])
        num_questions = st.slider("Nombre de questions :", min_value=1, max_value=10, value=5)
        temperature = st.slider(
            "Créativité / Température LLM (0.0=Factuel, 1.0=Créatif)", 
            min_value=0.0, max_value=1.0, value=0.7, step=0.05
        )
        st.session_state['temperature'] = temperature

        st.markdown("---")
        
        if st.button("Générer l'exam", type="primary"):
            if not text_source.strip():
                st.error("Veuillez fournir un texte source.")
            elif not LLM_CLIENT:
                 st.error("Erreur de Configuration. Veuillez vérifier votre clé API Groq (GROQ_API_KEY).")
            else:
                reset_all_data(full_reset=False)
                
                with st.spinner(f"Génération des questions de type {type_question} en cours"):
                    questions = generate_questions(text_source, difficulty, num_questions, type_question, temperature)
                    
                    if questions:
                        questions_dicts = []
                        for q in questions:
                            try:
                                qd = q.model_dump()
                            except Exception:
                                qd = q.__dict__
                            # Génération id pour chaque question
                            qid = hashlib.sha1((qd.get('question','') + qd.get('topic','')).encode()).hexdigest()[:8]
                            qd['id'] = qid
                            qd['__type__'] = type(q).__name__
                            questions_dicts.append(qd)
                        st.session_state['questions_data'] = questions_dicts
                        st.success(f"{len(questions)} questions de type '{type_question}' générées avec succès!")
                    else:
                        st.session_state['questions_data'] = None

    #Affichage exam
    if st.session_state['questions_data']:
        st.header("Quizz")
        questions_list = st.session_state['questions_data']
        
        # Prépare l'état des résultats avant l'affichage.
        results_snapshot_list = []
        results_snapshot = {}
        score = 0
        total = len(questions_list)
        
        if st.session_state['show_results']:
             score, total, results_snapshot_list = calculate_score(questions_list)
             results_snapshot = {r['index']: r for r in results_snapshot_list}
        
        # Initialisation stable des réponses dans st.session_state
        PLACEHOLDER_RADIO = "Choisir la bonne réponse"
        for i, question in enumerate(questions_list):
             qid = question.get('id')
             q_key = f"q_{qid}_answer"
             is_open = (question.get('__type__') == 'QuestionOuverte')
             if q_key not in st.session_state:
                 st.session_state[q_key] = "" if is_open else PLACEHOLDER_RADIO
        
        
        for i, question in enumerate(questions_list):
            q_index = i + 1
            display_question_test(
                question, 
                i, 
                st.session_state['show_results'], 
                results_snapshot.get(q_index)
            )

        st.markdown("---")
        
        col_buttons_1, col_buttons_2, _ = st.columns([0.3, 0.3, 0.4])
        
        with col_buttons_1:
            st.button("Vérifier réponses et correction", on_click=check_results, type="primary")

        with col_buttons_2:
            st.button("Recommencer l'exam", on_click=reset_show_results)
        
        #Affiche resultats
        if st.session_state['show_results']:
            
            if not results_snapshot:
                 score, total, results_snapshot_list = calculate_score(questions_list)
            
            st.header("Les resultaats")
            final_percentage = round((score/total)*100) if total > 0 else 0
            
            # Score
            st.success(f"### Score Final : {score}/{total} ({final_percentage}%)")
            st.markdown("---")
            
            #Tableau résumé reponses
            summary_data = []
            for result in results_snapshot_list:
                status_emoji = "✅ Correcte" if result['is_correct'] else ("❌ Fausse" if result['user_text'] != "Non répondu" else "❓ Non répondu")
                summary_data.append({
                    "Q.": result['index'],
                    "Thème": result['topic'],
                    "Statut": status_emoji
                })

            st.markdown("#### Résultats")
            st.dataframe(summary_data, use_container_width=True, hide_index=True)
            st.markdown("---")

            st.markdown("#### Correction")
            
            # Affichage correction
            for result in results_snapshot_list:
                col1, col2 = st.columns([0.8, 0.2])
                q_index = result['index']
                
                with col1:
                    st.markdown(f"**Q{q_index}.** {result['question']}")
                    st.markdown(f"**Votre réponse :** `{result['user_text']}`")
                    st.markdown(f"**Réponse correcte :** `{result['correct_text']}`")
                    st.markdown(f"*(Thème : {result['topic']})*")
                
                    # Affichage du feedback LLM pour les questions ouvertes corrigées
                    if result['llm_feedback']:
                        fb = result['llm_feedback']
                        st.markdown(f"**Score LLM :** **{fb['score_percentage']}%**")
                        st.info(f"**Feedback :** {fb['feedback_text']}")

                with col2:
                    if result['is_correct']:
                        st.success("Correcte")
                    elif result['user_text'] != "Non répondu":
                        st.error("Fausse")
                    else:
                        st.warning("Non Répondu")
                        
                st.markdown("---")

    # Debug: Bouton pour réinitialiser complètement l'application 
    if st.sidebar.button("Réinitialisation session", key='full_reset_btn'):
        reset_all_data(full_reset=True)
        st.rerun()


if __name__ == '__main__':
    main()