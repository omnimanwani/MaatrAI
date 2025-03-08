import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
import ssl
import joblib
import os
import pandas as pd
import numpy as np
import datetime
import joblib
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

ssl._create_default_https_context = ssl._create_unverified_context

class MaternalHealthChatbot:
    def __init__(self, model_path='pregnancy_risk_model.pkl', history_path='user_history.csv',
                neo4j_uri="neo4j+s://d130069a.databases.neo4j.io",
                neo4j_user="neo4j",
                neo4j_password="oKJNmzxrb2hxs0oysAAqfNq6ONXDRHmCP8M0fIYOc5k"):
        """
        Initialize the Maternal Health Chatbot with all necessary components

        Args:
            model_path: Path to the pre-trained risk prediction model
            history_path: Path to the user history CSV file
            neo4j_uri: URI for Neo4j database
            neo4j_user: Username for Neo4j database
            neo4j_password: Password for Neo4j database
        """
        # Load the risk prediction model
        try:
            self.risk_model_components = self.load_risk_analysis_model(model_path)
            print("Risk prediction model loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Risk prediction will be simulated.")
            self.risk_model_components = None

        # Initialize user history tracking
        self.history_path = history_path
        if not os.path.exists(history_path):
            # Create an empty history file with headers
            headers = ['date', 'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp',
                      'HeartRate', 'symptoms', 'risk_level', 'detected_conditions']
            pd.DataFrame(columns=headers).to_csv(history_path, index=False)
            print(f"Created new user history file at {history_path}")

        # Initialize Neo4j connection
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_driver = self._init_neo4j()

        # Initialize LLM
        self.llm, self.tokenizer = self._init_llm()

        # Initialize disease-symptom mapping
        self.disease_symptom_map = self._init_disease_symptom_mapping()

    def _init_disease_symptom_mapping(self):
        """
        Initialize the mapping between diseases and their symptoms

        Returns:
            Dictionary mapping diseases to their symptoms
        """
        # Comprehensive mapping of maternal health conditions to their symptoms
        mapping = {
            "Preeclampsia": ["headache", "swelling", "high blood pressure", "vision changes"],
            "Gestational Diabetes": ["high blood sugar", "frequent urination", "fatigue", "thirst"],
            "Placenta Previa": ["vaginal bleeding", "pain", "contractions"],
            "Anemia": ["fatigue", "weakness", "pale skin", "shortness of breath"],
            "HELLP Syndrome": ["nausea", "vomiting", "headache", "upper right abdominal pain"],
            "Hyperemesis Gravidarum": ["severe nausea", "vomiting", "dehydration", "weight loss"],
            "Preterm Labor": ["contractions", "pelvic pressure", "backache", "vaginal discharge"],
            "Ectopic Pregnancy": ["abdominal pain", "vaginal bleeding", "shoulder pain", "dizziness"],
            "Miscarriage": ["vaginal bleeding", "cramping", "tissue passing", "back pain"],
            "Deep Vein Thrombosis": ["leg swelling", "pain", "redness", "warmth"],
            "Urinary Tract Infection": ["painful urination", "frequent urination", "cloudy urine", "pelvic pain"],
            "Pregnancy-Induced Hypertension": ["high blood pressure", "headache", "vision changes", "swelling"],
            "Oligohydramnios": ["decreased fetal movement", "small uterus size", "fluid leakage"],
            "Polyhydramnios": ["shortness of breath", "swelling", "large uterus size", "discomfort"],
            "Cholestasis of Pregnancy": ["itching", "dark urine", "fatigue", "jaundice"],
            "Group B Strep Infection": ["fever", "chills", "urinary symptoms", "preterm labor"],
            "Placental Abruption": ["vaginal bleeding", "abdominal pain", "contractions", "back pain"],
            "Gestational Trophoblastic Disease": ["vaginal bleeding", "severe nausea", "high blood pressure", "no fetal movement"],
            "Cervical Insufficiency": ["painless cervical dilation", "preterm labor", "fetal membrane prolapse"],
            "Postpartum Hemorrhage": ["heavy bleeding", "dizziness", "low blood pressure", "rapid heart rate"]
        }

        # Create reverse mapping for efficient symptom lookup
        self.symptom_to_diseases = {}
        for disease, symptoms in mapping.items():
            for symptom in symptoms:
                if symptom not in self.symptom_to_diseases:
                    self.symptom_to_diseases[symptom] = []
                self.symptom_to_diseases[symptom].append(disease)

        return mapping

    def load_risk_analysis_model(self, file_path):
        """
        Load the saved model.

        Parameters:
            file_path: Path to the saved model

        Returns:
            dict: Model components
        """
        model_components = joblib.load(file_path)
        print("Model loaded successfully")
        return model_components

    def create_risk_analysis_interaction_features(self, X):
        """
        Create interaction and derived features from basic measurements.

        Parameters:
            X (DataFrame): Original features

        Returns:
            DataFrame: Original features plus engineered features
        """
        X_new = X.copy()

        # Blood pressure-related derivatives
        X_new['BP_Ratio'] = X['SystolicBP'] / X['DiastolicBP']
        X_new['Pulse_Pressure'] = X['SystolicBP'] - X['DiastolicBP']
        X_new['MAP'] = (X['SystolicBP'] + 2 * X['DiastolicBP']) / 3  # Mean Arterial Pressure

        # Age-related interactions
        X_new['Age_SystolicBP'] = X['Age'] * X['SystolicBP']
        X_new['Age_BS'] = X['Age'] * X['BS']

        # BS-related interactions
        X_new['BS_SystolicBP'] = X['BS'] * X['SystolicBP']

        # Squared terms for non-linear relationships
        X_new['BS_squared'] = X['BS'] ** 2
        X_new['Age_squared'] = X['Age'] ** 2

        return X_new

    def predict_risk(self, patient_data):
        """
        Predict risk level for a new patient using the loaded model components.

        Parameters:
            patient_data: Dictionary with patient features

        Returns:
            tuple: Predicted risk level and probabilities
        """
        if self.risk_model_components is None:
            return "Model not found"
        else:
            # Extract components
            model = self.risk_model_components['model']
            scaler = self.risk_model_components['scaler']
            selector = self.risk_model_components['selector']
            risk_mapping = self.risk_model_components['risk_mapping']

            # Create DataFrame for the new patient
            patient_df = pd.DataFrame([patient_data])

            # Engineer features
            patient_df_eng = self.create_risk_analysis_interaction_features(patient_df)

            # Select features
            patient_selected = selector.transform(patient_df_eng)

            # Scale the features
            patient_scaled = scaler.transform(patient_selected)

            # Make prediction
            risk_level_encoded = model.predict(patient_scaled)[0]
            risk_probs = model.predict_proba(patient_scaled)[0]

            # Map back to original risk level
            predicted_risk = risk_mapping[risk_level_encoded]

            return predicted_risk, risk_probs


    def _init_neo4j(self, uri="neo4j+s://c9dfed5f.databases.neo4j.io", user="neo4j", password="LDxuGG07tjf69oGO7rABpNPUZCy7hLFORGD3frCAzCc"):
          """
          Initialize the Neo4j database with the knowledge graph schema.
          This creates Disease, Symptom, Action nodes, Risk Level nodes, and their relationships.
          """
          driver = GraphDatabase.driver(uri, auth=(user, password))
          with driver.session() as session:
              # Clear existing data
              session.run("MATCH (n) DETACH DELETE n")

              # Create Risk Level nodes
              session.run("""
              CREATE (lowRisk:RiskLevel {name: 'LowRisk', description: 'Normal maternal health indicators'}),
                    (midRisk:RiskLevel {name: 'MidRisk', description: 'Some concerning maternal health indicators'}),
                    (highRisk:RiskLevel {name: 'HighRisk', description: 'Multiple concerning maternal health indicators'})
              """)

              # Create Disease nodes
              session.run("""
              CREATE (preeclampsia:Disease {name: 'preeclampsia', description: 'Pregnancy complication characterized by high blood pressure'}),
                    (gestational_diabetes:Disease {name: 'gestational_diabetes', description: 'High blood sugar during pregnancy'}),
                    (placenta_previa:Disease {name: 'placenta_previa', description: 'Vaginal bleeding, pain, contractions'}),
                    (anemia:Disease {name: 'anemia', description: 'Fatigue, weakness, pale skin, shortness of breath'}),
                    (hellp_syndrome:Disease {name: 'hellp_syndrome', description: 'Nausea, vomiting, headache, upper right abdominal pain'}),
                    (hyperemesis_gravidarum:Disease {name: 'hyperemesis_gravidarum', description: 'Severe nausea, vomiting, dehydration, weight loss'}),
                    (preterm_labor:Disease {name: 'preterm_labor', description: 'Contractions, pelvic pressure, backache, vaginal discharge'}),
                    (ectopic_pregnancy:Disease {name: 'ectopic_pregnancy', description: 'Abdominal pain, vaginal bleeding, shoulder pain, dizziness'}),
                    (miscarriage:Disease {name: 'miscarriage', description: 'Vaginal bleeding, cramping, tissue passing, back pain'}),
                    (deep_vein_thrombosis:Disease {name: 'deep_vein_thrombosis', description: 'Leg swelling, pain, redness, warmth'}),
                    (uti:Disease {name: 'uti', description: 'Painful urination, frequent urination, cloudy urine, pelvic pain'}),
                    (pregnancy_induced_hypertension:Disease {name: 'pregnancy_induced_hypertension', description: 'High blood pressure, headache, vision changes, swelling'}),
                    (oligohydramnios:Disease {name: 'oligohydramnios', description: 'Decreased fetal movement, small uterus size, fluid leakage'}),
                    (polyhydramnios:Disease {name: 'polyhydramnios', description: 'Shortness of breath, swelling, large uterus size, discomfort'}),
                    (cholestasis_of_pregnancy:Disease {name: 'cholestasis_of_pregnancy', description: 'Itching, dark urine, fatigue, jaundice'}),
                    (group_b_strep_infection:Disease {name: 'group_b_strep_infection', description: 'Fever, chills, urinary symptoms, preterm labor'}),
                    (placental_abruption:Disease {name: 'placental_abruption', description: 'Vaginal bleeding, abdominal pain, contractions, back pain'}),
                    (gestational_trophoblastic_disease:Disease {name: 'gestational_trophoblastic_disease', description: 'Vaginal bleeding, severe nausea, high blood pressure, no fetal movement'}),
                    (cervical_insufficiency:Disease {name: 'cervical_insufficiency', description: 'Painless cervical dilation, preterm labor, fetal membrane prolapse'}),
                    (postpartum_hemorrhage:Disease {name: 'postpartum_hemorrhage', description: 'Heavy bleeding, dizziness, low blood pressure, rapid heart rate'})
              """)

              # Create Symptom nodes
              session.run("""
              CREATE (headache:Symptom {name: 'Headache', threshold: 'persistent'}),
                    (swelling:Symptom {name: 'Swelling', threshold: 'persistent'}),
                    (highBP:Symptom {name: 'High Blood Pressure', threshold: '140/90'}),
                    (visionChanges:Symptom {name: 'Vision Changes', threshold: 'blurring'}),
                    (fatigue:Symptom {name: 'Fatigue', threshold: 'constant'}),
                    (frequentUrination:Symptom {name: 'Frequent Urination', threshold: 'more than 8 times a day'}),
                    (pain:Symptom {name: 'Pain', threshold: 'persistent'}),
                    (cramping:Symptom {name: 'Cramping', threshold: 'severe'}),
                    (shortnessBreath:Symptom {name: 'Shortness of Breath', threshold: 'at rest'}),
                    (fever:Symptom {name: 'Fever', threshold: '38°C'}),
                    (dizziness:Symptom {name: 'Dizziness', threshold: 'frequent'}),
                    (elevatedHR:Symptom {name: 'Elevated Heart Rate', threshold: 'above 100 bpm'}),
                    (nausea:Symptom {name: 'Nausea', threshold: 'persistent'}),
                    (abdominalPain:Symptom {name: 'Abdominal Pain', threshold: 'sharp or cramping'}),
                    (backPain:Symptom {name: 'Back Pain', threshold: 'persistent'}),
                    (vaginalBleeding:Symptom {name: 'Vaginal Bleeding', threshold: 'moderate to heavy'}),
                    (cloudyUrine:Symptom {name: 'Cloudy Urine', threshold: 'persistent'})
              """)

              # Create Action nodes
              session.run("""
              CREATE (contactHCP:Action {name: 'Contact healthcare provider immediately'}),
                    (monitorBP:Action {name: 'Monitor blood pressure every 2-4 hours'}),
                    (reduceSalt:Action {name: 'Reduce sodium intake and stay hydrated'}),
                    (restLying:Action {name: 'Rest lying on your left side to improve circulation'}),
                    (scheduleCheckup:Action {name: 'Schedule a check-up within 48 hours'}),
                    (monitorVitals:Action {name: 'Monitor your vital signs twice daily'}),
                    (ensureRest:Action {name: 'Ensure adequate rest and sleep'}),
                    (balancedDiet:Action {name: 'Maintain balanced diet and regular exercise'}),
                    (monitorSugar:Action {name: 'Monitor blood sugar levels before and after meals'}),
                    (trackVitals:Action {name: 'Track vital signs daily'}),
                    (increaseFluid:Action {name: 'Increase fluid intake to prevent dehydration'}),
                    (checkTemp:Action {name: 'Check temperature every 4-6 hours'})
              """)

              # Create relationships between diseases, symptoms, actions, and risk levels
              disease_symptom_data = {
                  "preeclampsia": ["Headache", "Swelling", "High Blood Pressure", "Vision Changes"],
                  "gestational_diabetes": ["High Blood Sugar", "Frequent Urination", "Fatigue", "Thirst"],
                  "placenta_previa": ["Vaginal Bleeding", "Pain", "Contractions"],
                  "anemia": ["Fatigue", "Weakness", "Pale Skin", "Shortness of Breath"],
                  "hellp_syndrome": ["Nausea", "Vomiting", "Headache", "Upper Right Abdominal Pain"],
                  "hyperemesis_gravidarum": ["Severe Nausea", "Vomiting", "Dehydration", "Weight Loss"],
                  "preterm_labor": ["Contractions", "Pelvic Pressure", "Backache", "Vaginal Discharge"],
                  "ectopic_pregnancy": ["Abdominal Pain", "Vaginal Bleeding", "Shoulder Pain", "Dizziness"],
                  "miscarriage": ["Vaginal Bleeding", "Cramping", "Tissue Passing", "Back Pain"],
                  "deep_vein_thrombosis": ["Leg Swelling", "Pain", "Redness", "Warmth"],
                  "uti": ["Painful Urination", "Frequent Urination", "Cloudy Urine", "Pelvic Pain"],
                  "pregnancy_induced_hypertension": ["High Blood Pressure", "Headache", "Vision Changes", "Swelling"],
                  "oligohydramnios": ["Decreased Fetal Movement", "Small Uterus Size", "Fluid Leakage"],
                  "polyhydramnios": ["Shortness of Breath", "Swelling", "Large Uterus Size", "Discomfort"],
                  "cholestasis_of_pregnancy": ["Itching", "Dark Urine", "Fatigue", "Jaundice"],
                  "group_b_strep_infection": ["Fever", "Chills", "Urinary Symptoms", "Preterm Labor"],
                  "placental_abruption": ["Vaginal Bleeding", "Abdominal Pain", "Contractions", "Back Pain"],
                  "gestational_trophoblastic_disease": ["Vaginal Bleeding", "Severe Nausea", "High Blood Pressure", "No Fetal Movement"],
                  "cervical_insufficiency": ["Painless Cervical Dilation", "Preterm Labor", "Fetal Membrane Prolapse"],
                  "postpartum_hemorrhage": ["Heavy Bleeding", "Dizziness", "Low Blood Pressure", "Rapid Heart Rate"]
              }

              # Create Disease-Symptom relationships
              for disease, symptoms in disease_symptom_data.items():
                  for symptom in symptoms:
                      session.run(f"""
                      MATCH (d:Disease), (s:Symptom)
                      WHERE d.name = '{disease}' AND s.name = '{symptom}'
                      CREATE (d)-[:HAS_SYMPTOM]->(s)
                      """)

              # Assign Diseases to Risk Levels
              risk_disease_mapping = {
                  "LowRisk": ["placenta_previa", "anemia", "oligohydramnios", "polyhydramnios"],
                  "MidRisk": ["preeclampsia", "gestational_diabetes", "hyperemesis_gravidarum", "preterm_labor", "miscarriage", "uti"],
                  "HighRisk": ["hellp_syndrome", "ectopic_pregnancy", "deep_vein_thrombosis", "pregnancy_induced_hypertension", "cholestasis_of_pregnancy", "group_b_strep_infection", "placental_abruption", "gestational_trophoblastic_disease", "cervical_insufficiency", "postpartum_hemorrhage"]
              }

              for risk_level, diseases in risk_disease_mapping.items():
                  for disease in diseases:
                      session.run(f"""
                      MATCH (d:Disease), (r:RiskLevel)
                      WHERE d.name = '{disease}' AND r.name = '{risk_level}'
                      CREATE (d)-[:HAS_RISK_LEVEL]->(r)
                      """)

              # Create Disease-Action relationships
              action_disease_mapping = {
                  "preeclampsia": [
                      "Contact healthcare provider immediately",
                      "Monitor blood pressure every 2-4 hours",
                      "Rest lying on your left side to improve circulation",
                      "Reduce sodium intake and stay hydrated",
                      "Schedule a check-up within 48 hours"
                  ],
                  "gestational_diabetes": [
                      "Monitor blood sugar levels before and after meals",
                      "Follow a diabetic-friendly diet plan",
                      "Exercise regularly to help control blood sugar",
                      "Track weight gain and discuss with healthcare provider",
                      "Schedule regular follow-ups to monitor glucose levels"
                  ],
                  "placenta_previa": [
                      "Schedule a check-up within 48 hours",
                      "Monitor your vital signs twice daily",
                      "Avoid heavy lifting or strenuous activities",
                      "Rest and keep pelvis elevated",
                      "Consider bed rest if advised by the doctor"
                  ],
                  "anemia": [
                      "Increase iron-rich foods like leafy greens, red meat, and legumes",
                      "Take iron supplements as prescribed",
                      "Monitor hemoglobin levels regularly",
                      "Avoid calcium-rich foods and supplements with iron intake",
                      "Consult with a healthcare provider for blood transfusion if needed"
                  ],
                  "hellp_syndrome": [
                      "Hospitalize immediately for monitoring",
                      "Administer corticosteroids to speed up lung development if premature delivery is expected",
                      "Manage blood pressure with antihypertensive medications",
                      "Monitor liver function tests and platelets",
                      "Prepare for potential early delivery if necessary"
                  ],
                  "hyperemesis_gravidarum": [
                      "Monitor hydration and manage electrolytes",
                      "Prescribe anti-nausea medication",
                      "Consider IV fluids for rehydration if vomiting is severe",
                      "Avoid trigger foods or smells",
                      "Rest and reduce stress"
                  ],
                  "preterm_labor": [
                      "Monitor contractions and timing with a contraction timer",
                      "Consult healthcare provider if contractions last more than an hour",
                      "Rest and stay hydrated",
                      "Consider steroid treatment to speed up fetal lung development",
                      "Prepare for early delivery by planning hospital arrangements"
                  ],
                  "ectopic_pregnancy": [
                      "Seek immediate medical attention for severe abdominal pain",
                      "Monitor for signs of shock and heavy bleeding",
                      "Administer methotrexate or surgical intervention to stop the pregnancy",
                      "Prepare for possible surgical removal of the fallopian tube",
                      "Follow-up care to monitor HCG levels"
                  ],
                  "miscarriage": [
                      "Seek medical evaluation for bleeding or cramping",
                      "Rest and avoid heavy physical activity",
                      "Use pain relief like acetaminophen as advised by the doctor",
                      "Prepare emotionally and consult with a counselor if needed",
                      "Follow up with ultrasound to confirm the status of the pregnancy"
                  ],
                  "deep_vein_thrombosis": [
                      "Wear compression stockings to reduce swelling",
                      "Take prescribed blood thinners",
                      "Avoid long periods of immobility and move legs frequently",
                      "Elevate legs above the heart level to reduce swelling",
                      "Monitor for signs of embolism and seek immediate help if needed"
                  ],
                  "uti": [
                      "Drink plenty of fluids to flush out bacteria",
                      "Take prescribed antibiotics as directed",
                      "Avoid bladder irritants like caffeine and alcohol",
                      "Use a heating pad to relieve discomfort",
                      "Schedule follow-up appointments to ensure infection is resolved"
                  ],
                  "pregnancy_induced_hypertension": [
                      "Monitor blood pressure regularly",
                      "Avoid stress and rest in a comfortable position",
                      "Take prescribed antihypertensive medications",
                      "Track swelling in feet and hands and report to a doctor",
                      "Hospitalization for severe cases and close monitoring"
                  ],
                  "oligohydramnios": [
                      "Increase fluid intake to prevent dehydration",
                      "Monitor fetal movement regularly",
                      "Consider bed rest to avoid further decrease in amniotic fluid",
                      "Consider amnioinfusion during labor if necessary",
                      "Follow-up ultrasound to check amniotic fluid levels"
                  ],
                  "polyhydramnios": [
                      "Monitor fetal heart rate and amniotic fluid levels",
                      "Avoid excessive fluid intake",
                      "Rest and reduce physical activity",
                      "Consider amniocentesis for further evaluation if needed",
                      "Follow up with regular check-ups and ultrasounds"
                  ],
                  "cholestasis_of_pregnancy": [
                      "Monitor liver function tests regularly",
                      "Take prescribed ursodeoxycholic acid to manage symptoms",
                      "Use antihistamines for itching relief",
                      "Rest and reduce stress",
                      "Prepare for possible early delivery if symptoms worsen"
                  ],
                  "group_b_strep_infection": [
                      "Administer antibiotics during labor to reduce risk of neonatal infection",
                      "Monitor for signs of infection such as fever and chills",
                      "Check for urinary symptoms and treat infections promptly",
                      "Ensure early testing for group B strep at 36 weeks",
                      "Consult with a doctor for any signs of preterm labor or infection"
                  ],
                  "placental_abruption": [
                      "Seek immediate medical attention for vaginal bleeding and abdominal pain",
                      "Monitor fetal heart rate and maternal condition",
                      "Prepare for potential early delivery by scheduling C-section if necessary",
                      "Provide blood transfusions if required",
                      "Avoid any physical activity and monitor signs of shock"
                  ],
                  "gestational_trophoblastic_disease": [
                      "Monitor hCG levels regularly to track the progression of the disease",
                      "Consider surgical removal of the abnormal tissue",
                      "Prepare for chemotherapy if malignant tissue is found",
                      "Follow up with ultrasounds to check for recurrence",
                      "Consult with a fertility specialist for future pregnancies"
                  ],
                  "cervical_insufficiency": [
                      "Consider a cerclage procedure to close the cervix",
                      "Monitor for signs of preterm labor and seek medical help immediately",
                      "Avoid heavy physical activity and prolonged standing",
                      "Rest and reduce stress",
                      "Follow up with regular cervical length assessments"
                  ],
                  "postpartum_hemorrhage": [
                      "Apply uterine massage to stimulate contraction",
                      "Administer uterotonic drugs like oxytocin to control bleeding",
                      "Ensure IV access for fluid resuscitation",
                      "Consider blood transfusions if necessary",
                      "Monitor for signs of shock and manage promptly"
                  ]
              }


              for disease, actions in action_disease_mapping.items():
                  for action in actions:
                      session.run(f"""
                      MATCH (d:Disease), (a:Action)
                      WHERE d.name = '{disease}' AND a.name = '{action}'
                      CREATE (d)-[:REQUIRES_ACTION]->(a)
                      """)

          driver.close()
          print("Neo4j database initialized successfully.")

    def _init_llm(self):
        """Initialize LLM model for conversation"""
        try:
            # For real implementation, use a more capable model
            # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            # For demonstration, using a smaller model that's easier to run locally
            model_name = "gpt2"  # In production, replace with Mistral-7B or Llama 2

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"LLM model {model_name} loaded successfully.")
            return model, tokenizer
        except Exception as e:
            print(f"Warning: Could not load LLM model. Will use template responses. Error: {e}")
            return None, None

    def collect_vitals(self, vitals):
        """
        Collect and validate user vital signs

        Args:
            vitals: Dictionary containing user vital measurements

        Returns:
            Processed vitals with any needed feature engineering
        """
        required_fields = ['SystolicBP', 'DiastolicBP', 'BS', 'HeartRate', 'BodyTemp', 'Age']

        # Validate required input
        for field in required_fields:
            if field not in vitals:
                raise ValueError(f"Missing required vital measurement: {field}")

        # Add date to the data
        vitals_with_date = vitals.copy()
        vitals_with_date['date'] = datetime.datetime.now().strftime("%Y-%m-%d")

        return vitals_with_date

    def parse_symptoms(self, symptoms_text):
        """
        Parse free-text symptoms into structured format

        Args:
            symptoms_text: String containing user-reported symptoms

        Returns:
            List of identified symptoms
        """
        if not symptoms_text:
            return []

        # Tokenize and normalize symptoms text
        tokens = word_tokenize(symptoms_text.lower())
        stops = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stops and token.isalpha()]

        # Extract symptoms using n-grams matching against known symptoms
        identified_symptoms = []

        # Get all possible symptoms from our mapping
        all_symptoms = set()
        for symptoms in self.disease_symptom_map.values():
            all_symptoms.update(symptoms)

        # Check for exact matches with symptoms in our mapping
        for n in range(3, 0, -1):  # Try 3-grams, then 2-grams, then 1-grams
            if n > len(tokens):
                continue

            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                if ngram in all_symptoms and ngram not in identified_symptoms:
                    identified_symptoms.append(ngram)

        # Check for partial matches
        if not identified_symptoms:
            for symptom in all_symptoms:
                symptom_tokens = set(word_tokenize(symptom.lower()))
                if any(token in symptom_tokens for token in tokens):
                    identified_symptoms.append(symptom)

        return identified_symptoms

    def predict_diseases(self, symptoms, vitals):
        """
        Predict potential diseases based on symptoms and vitals

        Args:
            symptoms: List of identified symptoms
            vitals: Dictionary of user vital measurements

        Returns:
            List of potential diseases with confidence scores
        """
        if not symptoms:
            return []

        # Count disease matches based on symptoms
        disease_matches = {}
        for symptom in symptoms:
            if symptom in self.symptom_to_diseases:
                for disease in self.symptom_to_diseases[symptom]:
                    if disease not in disease_matches:
                        disease_matches[disease] = 0
                    disease_matches[disease] += 1

        # Calculate confidence score based on number of matching symptoms
        results = []
        for disease, match_count in disease_matches.items():
            total_symptoms = len(self.disease_symptom_map[disease])
            confidence = match_count / total_symptoms
            results.append({
                "disease": disease,
                "confidence": confidence,
                "matching_symptoms": match_count,
                "total_symptoms": total_symptoms
            })

        # Add vitals-based conditions
        systolic = vitals.get('SystolicBP', 120)
        diastolic = vitals.get('DiastolicBP', 80)
        bs = vitals.get('BS', 5.0)

        # Check for preeclampsia based on blood pressure
        if systolic > 140 or diastolic > 90:
            preeclampsia_exists = any(r["disease"] == "Preeclampsia" for r in results)
            if not preeclampsia_exists:
                results.append({
                    "disease": "Preeclampsia",
                    "confidence": 0.7,
                    "matching_symptoms": 1,
                    "total_symptoms": 4,
                    "note": "Added based on high blood pressure"
                })
            else:
                # Boost confidence for existing preeclampsia
                for r in results:
                    if r["disease"] == "Preeclampsia":
                        r["confidence"] = max(r["confidence"], 0.7)
                        r["note"] = "Confidence boosted due to high blood pressure"

        # Check for gestational diabetes based on blood sugar
        if bs > 7.8:  # 140 mg/dL
            gd_exists = any(r["disease"] == "Gestational Diabetes" for r in results)
            if not gd_exists:
                results.append({
                    "disease": "Gestational Diabetes",
                    "confidence": 0.7,
                    "matching_symptoms": 1,
                    "total_symptoms": 4,
                    "note": "Added based on high blood sugar"
                })
            else:
                # Boost confidence for existing gestational diabetes
                for r in results:
                    if r["disease"] == "Gestational Diabetes":
                        r["confidence"] = max(r["confidence"], 0.7)
                        r["note"] = "Confidence boosted due to high blood sugar"

        # Sort by confidence score (descending)
        results.sort(key=lambda x: x["confidence"], reverse=True)

        # Return top 3 predictions with confidence > 0.2
        return [r for r in results if r["confidence"] > 0.2][:3]

    def save_user_data(self, vitals, symptoms_text, risk_level, detected_conditions):
        """
        Save user data to history file

        Args:
            vitals: Dictionary containing user vital measurements
            symptoms_text: String of user-reported symptoms
            risk_level: Predicted risk level
            detected_conditions: List of predicted diseases

        Returns:
            None
        """
        # Prepare data row
        data = {
            'date': vitals.get('date', datetime.datetime.now().strftime("%Y-%m-%d")),
            'Age': vitals.get('Age', None),
            'SystolicBP': vitals.get('SystolicBP', None),
            'DiastolicBP': vitals.get('DiastolicBP', None),
            'BS': vitals.get('BS', None),
            'BodyTemp': vitals.get('BodyTemp', None),
            'HeartRate': vitals.get('HeartRate', None),
            'symptoms': symptoms_text,
            'risk_level': risk_level,
            'detected_conditions': ','.join([d["disease"] for d in detected_conditions]) if detected_conditions else 'N/A'
        }

        # Append to CSV
        df = pd.DataFrame([data])
        df.to_csv(self.history_path, mode='a', header=False, index=False)
        print(f"User data saved to {self.history_path}")

    def analyze_health_trends(self):
        """
        Analyze user health history for concerning trends.

        Returns:
            Dictionary of detected health concerns and descriptions.
        """
        try:
            # Load history
            history = pd.read_csv(self.history_path)

            # Filter to last 5 days
            history['date'] = pd.to_datetime(history['date'])
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=5)
            recent_history = history[history['date'] >= cutoff_date]

            if len(recent_history) == 0:
                return {"message": "Not enough historical data to analyze trends."}

            concerns = {}

            # Check for diseases and related symptoms

            # Preeclampsia: headache, swelling, high blood pressure, vision changes
            preeclampsia_symptoms = ['headache', 'swelling', 'high blood pressure', 'vision changes']
            if any(symptom in recent_history.columns for symptom in preeclampsia_symptoms):
                high_bp_days = recent_history[(recent_history['SystolicBP'] > 140) |
                                              (recent_history['DiastolicBP'] > 90)]
                if len(high_bp_days) >= 2:
                    concerns["preeclampsia"] = {
                        "description": "Blood pressure has been elevated for multiple days",
                        "severity": "high",
                        "data_points": len(high_bp_days)
                    }

            # Gestational Diabetes: high blood sugar, frequent urination, fatigue, thirst
            if 'BS' in recent_history.columns and any(symptom in recent_history.columns for symptom in ['frequent urination', 'fatigue', 'thirst']):
                high_bs_days = recent_history[recent_history['BS'] > 7.8]
                if len(high_bs_days) >= 4:
                    concerns["gestational_diabetes"] = {
                        "description": "Blood sugar has been consistently elevated",
                        "severity": "mid",
                        "data_points": len(high_bs_days)
                    }

            # Placenta Previa: vaginal bleeding, pain, contractions
            if any(symptom in recent_history.columns for symptom in ['vaginal bleeding', 'pain', 'contractions']):
                concerns["placenta_previa"] = {
                    "description": "Symptoms of placenta previa detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Anemia: fatigue, weakness, pale skin, shortness of breath
            if any(symptom in recent_history.columns for symptom in ['fatigue', 'weakness', 'pale skin', 'shortness of breath']):
                concerns["anemia"] = {
                    "description": "Symptoms of anemia detected",
                    "severity": "mid",
                    "data_points": len(recent_history)
                }

            # HELLP Syndrome: nausea, vomiting, headache, upper right abdominal pain
            if any(symptom in recent_history.columns for symptom in ['nausea', 'vomiting', 'headache', 'upper right abdominal pain']):
                concerns["hellp_syndrome"] = {
                    "description": "Symptoms of HELLP syndrome detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Hyperemesis Gravidarum: severe nausea, vomiting, dehydration, weight loss
            if any(symptom in recent_history.columns for symptom in ['severe nausea', 'vomiting', 'dehydration', 'weight loss']):
                concerns["hyperemesis_gravidarum"] = {
                    "description": "Symptoms of hyperemesis gravidarum detected",
                    "severity": "mid",
                    "data_points": len(recent_history)
                }

            # Preterm Labor: contractions, pelvic pressure, backache, vaginal discharge
            if any(symptom in recent_history.columns for symptom in ['contractions', 'pelvic pressure', 'backache', 'vaginal discharge']):
                concerns["preterm_labor"] = {
                    "description": "Signs of preterm labor detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Ectopic Pregnancy: abdominal pain, vaginal bleeding, shoulder pain, dizziness
            if any(symptom in recent_history.columns for symptom in ['abdominal pain', 'vaginal bleeding', 'shoulder pain', 'dizziness']):
                concerns["ectopic_pregnancy"] = {
                    "description": "Signs of ectopic pregnancy detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Miscarriage: vaginal bleeding, cramping, tissue passing, back pain
            if any(symptom in recent_history.columns for symptom in ['vaginal bleeding', 'cramping', 'tissue passing', 'back pain']):
                concerns["miscarriage"] = {
                    "description": "Symptoms of miscarriage detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Deep Vein Thrombosis: leg swelling, pain, redness, warmth
            if any(symptom in recent_history.columns for symptom in ['leg swelling', 'pain', 'redness', 'warmth']):
                concerns["deep_vein_thrombosis"] = {
                    "description": "Signs of deep vein thrombosis detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Urinary Tract Infection: painful urination, frequent urination, cloudy urine, pelvic pain
            if any(symptom in recent_history.columns for symptom in ['painful urination', 'frequent urination', 'cloudy urine', 'pelvic pain']):
                concerns["urinary_tract_infection"] = {
                    "description": "Symptoms of urinary tract infection detected",
                    "severity": "mid",
                    "data_points": len(recent_history)
                }

            # Pregnancy-Induced Hypertension: high blood pressure, headache, vision changes, swelling
            if any(symptom in recent_history.columns for symptom in ['high blood pressure', 'headache', 'vision changes', 'swelling']):
                high_bp_days = recent_history[(recent_history['SystolicBP'] > 140) |
                                              (recent_history['DiastolicBP'] > 90)]
                if len(high_bp_days) >= 2:
                    concerns["pregnancy_induced_hypertension"] = {
                        "description": "Elevated blood pressure detected",
                        "severity": "high",
                        "data_points": len(high_bp_days)
                    }

            # Oligohydramnios: decreased fetal movement, small uterus size, fluid leakage
            if any(symptom in recent_history.columns for symptom in ['decreased fetal movement', 'small uterus size', 'fluid leakage']):
                concerns["oligohydramnios"] = {
                    "description": "Signs of oligohydramnios detected",
                    "severity": "mid",
                    "data_points": len(recent_history)
                }

            # Polyhydramnios: shortness of breath, swelling, large uterus size, discomfort
            if any(symptom in recent_history.columns for symptom in ['shortness of breath', 'swelling', 'large uterus size', 'discomfort']):
                concerns["polyhydramnios"] = {
                    "description": "Signs of polyhydramnios detected",
                    "severity": "mid",
                    "data_points": len(recent_history)
                }

            # Cholestasis of Pregnancy: itching, dark urine, fatigue, jaundice
            if any(symptom in recent_history.columns for symptom in ['itching', 'dark urine', 'fatigue', 'jaundice']):
                concerns["cholestasis_of_pregnancy"] = {
                    "description": "Symptoms of cholestasis of pregnancy detected",
                    "severity": "mid",
                    "data_points": len(recent_history)
                }

            # Group B Strep Infection: fever, chills, urinary symptoms, preterm labor
            if any(symptom in recent_history.columns for symptom in ['fever', 'chills', 'urinary symptoms', 'preterm labor']):
                concerns["group_b_strep_infection"] = {
                    "description": "Signs of Group B strep infection detected",
                    "severity": "mid",
                    "data_points": len(recent_history)
                }

            # Placental Abruption: vaginal bleeding, abdominal pain, contractions, back pain
            if any(symptom in recent_history.columns for symptom in ['vaginal bleeding', 'abdominal pain', 'contractions', 'back pain']):
                concerns["placental_abruption"] = {
                    "description": "Symptoms of placental abruption detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Gestational Trophoblastic Disease: vaginal bleeding, severe nausea, high blood pressure, no fetal movement
            if any(symptom in recent_history.columns for symptom in ['vaginal bleeding', 'severe nausea', 'high blood pressure', 'no fetal movement']):
                concerns["gestational_trophoblastic_disease"] = {
                    "description": "Signs of gestational trophoblastic disease detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Cervical Insufficiency: painless cervical dilation, preterm labor, fetal membrane prolapse
            if any(symptom in recent_history.columns for symptom in ['painless cervical dilation', 'preterm labor', 'fetal membrane prolapse']):
                concerns["cervical_insufficiency"] = {
                    "description": "Signs of cervical insufficiency detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            # Postpartum Hemorrhage: heavy bleeding, dizziness, low blood pressure, rapid heart rate
            if any(symptom in recent_history.columns for symptom in ['heavy bleeding', 'dizziness', 'low blood pressure', 'rapid heart rate']):
                concerns["postpartum_hemorrhage"] = {
                    "description": "Signs of postpartum hemorrhage detected",
                    "severity": "high",
                    "data_points": len(recent_history)
                }

            return concerns

        except Exception as e:
            print(f"Error analyzing health trends: {e}")
            return {"error": "Could not analyze health trends due to data issue."}


        except Exception as e:
            print(f"Error analyzing health trends: {e}")
            return {"error": "Could not analyze health trends due to data issue."}

    def get_recommendations(self, risk_level, detected_conditions, concerns=None):
        """
        Get recommendations from the knowledge graph based on risk level, detected conditions, and concerns

        Args:
            risk_level: Predicted risk level (low, mid, high)
            detected_conditions: List of predicted diseases with confidence scores
            concerns: Dictionary of health concerns from trend analysis

        Returns:
            Dictionary with general and condition-specific recommendations
        """
        recommendations = {
            "general": [],
            "condition_specific": {}
        }

        # Format risk level to match Neo4j data
        neo4j_risk_level = risk_level.replace(" ", "").capitalize() + "Risk"

        # Try to get recommendations from Neo4j
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    # Query for general recommendations based on risk level
                    query = """
                    MATCH (d:Disease)-[:REQUIRES_ACTION]->(a:Action)
                    WHERE d.name = $risk_level
                    RETURN a.name AS action
                    """
                    result = session.run(query, risk_level=neo4j_risk_level)
                    for record in result:
                        recommendations["general"].append(record["action"])

                    # Get recommendations for detected conditions
                    for condition in detected_conditions:
                        disease_name = condition["disease"]
                        disease_key = disease_name.lower().replace(" ", "_")

                        query = """
                        MATCH (d:Disease)-[:REQUIRES_ACTION]->(a:Action)
                        WHERE d.name = $disease_name OR d.name = $disease_key
                        RETURN a.name AS action
                        """
                        result = session.run(query, disease_name=disease_name, disease_key=disease_key)
                        condition_recs = []
                        for record in result:
                            condition_recs.append(record["action"])

                        if condition_recs:
                            recommendations["condition_specific"][disease_name] = condition_recs

                    # If there are specific concerns, get more targeted recommendations
                    if concerns and len(concerns) > 0:
                        for concern, details in concerns.items():
                            if concern not in recommendations["condition_specific"] and concern not in ["message", "error"]:
                                concern_key = concern.lower().replace(" ", "_")
                                query = """
                                MATCH (d:Disease)-[:REQUIRES_ACTION]->(a:Action)
                                WHERE d.name = $concern_name OR d.name = $concern_key
                                RETURN a.name AS action
                                """
                                result = session.run(query, concern_name=concern, concern_key=concern_key)
                                concern_recs = []
                                for record in result:
                                    concern_recs.append(record["action"])

                                if concern_recs:
                                    recommendations["condition_specific"][concern] = concern_recs
            except Exception as e:
                print(f"Error querying Neo4j: {e}")
                # Fall back to default recommendations

        # If no recommendations or Neo4j is not available, use fallbacks
        if not recommendations["general"]:
            if "high" in risk_level.lower():
                recommendations["general"] = [
                    "Contact your healthcare provider immediately",
                    "Rest and monitor your blood pressure every 2 hours",
                    "Reduce sodium intake and stay hydrated"
                ]
            elif "mid" in risk_level.lower():
                recommendations["general"] = [
                    "Schedule a check-up within the next 48 hours",
                    "Monitor your vital signs twice daily",
                    "Ensure you're getting adequate rest"
                ]
            else:  # Low Risk
                recommendations["general"] = [
                    "Continue regular prenatal care schedule",
                    "Maintain a balanced diet and regular exercise",
                    "Track your vital signs daily"
                ]

        # Add fallback condition-specific recommendations if needed
        fallback_recommendations = {
            "Preeclampsia": [
                "Rest lying on your left side to improve circulation",
                "Monitor your blood pressure more frequently",
                "Reduce salt intake and stay hydrated"
            ],
            "Gestational Diabetes": [
                "Follow a diabetic-friendly diet plan",
                "Monitor blood sugar levels before and after meals",
                "Regular moderate exercise as approved by your doctor"
            ],
            "Anemia": [
                "Take prescribed iron supplements",
                "Eat iron-rich foods like spinach and beans",
                "Get plenty of vitamin C to improve iron absorption"
            ],
            "Placenta Previa": [
                "Avoid strenuous activities and heavy lifting",
                "Report any vaginal bleeding immediately",
                "Follow pelvic rest guidelines from your doctor"
            ],
            "HELLP Syndrome": [
                "Seek immediate medical attention",
                "Complete bed rest as advised by your doctor",
                "Monitor for worsening symptoms like pain or vision changes"
            ],
            "Infection": [
                "Check temperature every 4-6 hours",
                "Increase fluid intake to prevent dehydration",
                "Take prescribed antibiotics as directed"
            ]
        }

        # Add fallback recommendations for conditions that don't have Neo4j recommendations
        for condition in detected_conditions:
            disease_name = condition["disease"]
            if disease_name not in recommendations["condition_specific"] and disease_name in fallback_recommendations:
                recommendations["condition_specific"][disease_name] = fallback_recommendations[disease_name]

        # Add recommendations for concerns from trend analysis
        if concerns:
            for concern, details in concerns.items():
                if concern not in recommendations["condition_specific"] and concern not in ["message", "error"] and concern in fallback_recommendations:
                    recommendations["condition_specific"][concern] = fallback_recommendations[concern]

        return recommendations

    def generate_response(self, vitals, symptoms_text, risk_level, risk_probs,
                         detected_conditions, concerns, recommendations):
        """
        Generate a conversational response using the LLM

        Args:
            vitals: User's vital measurements
            symptoms_text: User's symptoms in text format
            risk_level: Predicted risk level
            risk_probs: Risk probabilities
            detected_conditions: List of detected diseases
            concerns: Dictionary of health concerns
            recommendations: Dictionary of recommendations

        Returns:
            Conversational response to the user
        """
        if self.llm and self.tokenizer:
            try:
                # Prepare context for the LLM
                conditions_text = ""
                if detected_conditions and len(detected_conditions) > 0:
                    conditions_text = "Based on your symptoms and vitals, you may have: "
                    for condition in detected_conditions:
                        conditions_text += f"{condition['disease']} (confidence: {condition['confidence']:.0%}), "
                    conditions_text = conditions_text.rstrip(", ") + ". "

                concerns_text = ""
                if concerns and len(concerns) > 0 and not ("message" in concerns and len(concerns) == 1):
                    concerns_text = "I've also noticed some potential concerns from your history: "
                    for concern, details in concerns.items():
                        if concern not in ["message", "error"]:
                            concerns_text += f"{details['description']}. "

                recommendations_text = "Here are my general recommendations: " + ", ".join(recommendations["general"]) + ". "

                if recommendations["condition_specific"]:
                    recommendations_text += "For your specific conditions: "
                    for condition, recs in recommendations["condition_specific"].items():
                        if recs:
                            recommendations_text += f"For {condition}: {', '.join(recs)}. "

                # Format risk probabilities
                if isinstance(risk_probs, list):
                    risk_prob_text = f"(Low: {risk_probs[0]:.1%}, Medium: {risk_probs[1]:.1%}, High: {risk_probs[2]:.1%})"
                else:
                    risk_prob_text = ""
# Create prompt for the LLM
                prompt = f"""
                As a maternal health assistant, respond to a pregnant woman with the following health information:
                Blood Pressure: {vitals.get('SystolicBP', 'N/A')}/{vitals.get('DiastolicBP', 'N/A')}
                Blood Sugar: {vitals.get('BS', 'N/A')} mmol/L
                Heart Rate: {vitals.get('HeartRate', 'N/A')} bpm
                Temperature: {vitals.get('BodyTemp', 'N/A')} °C
                Symptoms: {symptoms_text}

                Risk Assessment: {risk_level} {risk_prob_text}
                {conditions_text}
                {concerns_text}

                {recommendations_text}

                Provide a caring, supportive response that explains the findings and recommendations in simple terms.
                If the risk level is high or any severe conditions are detected, emphasize the importance of contacting healthcare providers immediately.
                """

                # Generate response with the LLM
                inputs = self.tokenizer(prompt, return_tensors="pt")
                output = self.llm.generate(
                    inputs.input_ids,
                    max_length=500,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

                # Decode the response
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)

                # Extract only the chatbot's response (not the prompt)
                response_only = response.split("immediately.")[-1].strip()

                return response_only

            except Exception as e:
                print(f"Error generating LLM response: {e}")
                # Fall back to template response

        # Fallback templated response if LLM fails
        if "high" in risk_level.lower():
            severity_message = "Your current readings indicate a high-risk situation that requires immediate attention."
            action_message = "Please contact your healthcare provider right away."
        elif "mid" in risk_level.lower():
            severity_message = "Your readings show some potential concerns that should be addressed soon."
            action_message = "Please schedule an appointment with your healthcare provider within the next 48 hours."
        else:
            severity_message = "Your readings are generally within normal ranges, which is good news."
            action_message = "Continue with your regular prenatal care schedule."

        # Create condition-specific messages
        conditions_message = ""
        if detected_conditions and len(detected_conditions) > 0:
            conditions_message = "Based on your symptoms and measurements, you may be experiencing: "
            for condition in detected_conditions:
                conditions_message += f"{condition['disease']}, "
            conditions_message = conditions_message.rstrip(", ") + ". "

        # Create recommendations message
        recs_message = "My recommendations: " + ", ".join(recommendations["general"]) + ". "

        if recommendations["condition_specific"]:
            for condition, recs in recommendations["condition_specific"].items():
                if recs:
                    recs_message += f"For {condition}, please: {', '.join(recs)}. "

        # Assemble the response
        template_response = f"""
        Thank you for sharing your health information with me.

        {severity_message} {conditions_message}

        {recs_message}

        {action_message}

        Remember to keep tracking your vital signs daily and report any sudden changes to your healthcare provider.
        """

        return template_response.strip()

    def process_user_input(self, vitals_dict, symptoms_text):
        """
        Process user input to analyze health status and generate recommendations

        Args:
            vitals_dict: Dictionary with user vital measurements
            symptoms_text: String containing user-reported symptoms

        Returns:
            Conversational response with health assessment and recommendations
        """
        try:
            # Step 1: Collect and validate vitals
            vitals = self.collect_vitals(vitals_dict)

            # Step 2: Parse symptoms
            identified_symptoms = self.parse_symptoms(symptoms_text)

            # Step 3: Predict risk level
            vitals_without_date = {k: vitals[k] for k in ['Age','SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']}
            risk_level, risk_probs = self.predict_risk(vitals_without_date)

            # Step 4: Predict potential diseases
            detected_conditions = self.predict_diseases(identified_symptoms, vitals)

            # Step 5: Save user data
            self.save_user_data(vitals, symptoms_text, risk_level, detected_conditions)

            # Step 6: Analyze trends in user history
            concerns = self.analyze_health_trends()

            # Step 7: Get recommendations
            recommendations = self.get_recommendations(risk_level, detected_conditions, concerns)

            # Step 8: Generate conversational response
            response = self.generate_response(
                vitals, symptoms_text, risk_level, risk_probs,
                detected_conditions, concerns, recommendations
            )

            # Prepare result summary for logging and UI
            result = {
                "risk_level": risk_level,
                "detected_conditions": detected_conditions,
                "concerns": concerns,
                "recommendations": recommendations,
                "response": response
            }

            return result

        except Exception as e:
            error_msg = f"Error processing input: {e}"
            print(error_msg)
            return {
                "error": error_msg,
                "response": "I'm sorry, I encountered an error while processing your information. Please make sure all required measurements are provided and try again."
            }

    def close(self):
        """Close any open connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            print("Neo4j connection closed.")
