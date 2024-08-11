from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model and scaler
model = joblib.load('lgbm_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the expected features (all except 'InscClaimAmtReimbursed')
EXPECTED_FEATURES = [
    'BeneID', 'ClaimID', 'Provider', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician',
    'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
    'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
    'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
    'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
    'DeductibleAmtPaid', 'ClmAdmitDiagnosisCode', 'Gender', 'Race', 'RenalDiseaseIndicator',
    'State', 'County', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'ChronicCond_Alzheimer',
    'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
    'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes',
    'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
    'ChronicCond_stroke', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'IsDeceased', 'Age', 'ClaimDuration',
    'AgeAtClaim', 'ClaimLengthCategory', 'AgeCategory', 'NumChronicConditions', 'NumDiagnoses',
    'NumProcedures', 'AgeChronicInteraction', 'DurationProcedureInteraction'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.json['features']
        
        # Ensure all required features are present
        if not all(feature in data for feature in EXPECTED_FEATURES):
            missing_features = [f for f in EXPECTED_FEATURES if f not in data]
            return jsonify({'error': f'Missing required features: {", ".join(missing_features)}'}), 400
        
        # Create feature array, converting each feature to float
        features = []
        for feature in EXPECTED_FEATURES:
            try:
                features.append(float(data[feature]))
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}. Expected float.'}), 400
        
        # Convert features to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        return jsonify({'prediction': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
